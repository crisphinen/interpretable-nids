# nesy-nids: differentiable rule learning for iot network intrusion detection.
#
# architecture:
# nesynids = rulebank (differentiable logical rules) + neuralfallback (mlp) + gated mixing.
#
# rule form: score_r = product_i sigmoid(k*(x[i]-theta_i)) for 'gt' conditions (x > theta)
# and product_j sigmoid(k*(theta_j - x[j])) for 'lt' conditions (x < theta).
# training uses k-annealing: k starts at 1 (soft), grows to 10 (hard, near-binary gates).
# high k makes gates approach step functions so rules become crisp logical statements.
#
# ood scoring: max rule activation per sample (rule confidence).
# low confidence = sample fits no known rule = potentially ood.

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# rule template: describes the structure of a single logical rule.
@dataclass
class RuleTemplate:
    name: str
    feature_indices: List[int]              # which features participate
    condition_types: List[Literal['gt','lt']]  # 'gt' = x>theta, 'lt' = x<theta
    init_thresholds: List[float]            # initial threshold values


# ctu-iot-23 rule templates
# features (idx): IAT=6, Rate=10, syn_count=21, duration=23, resp_pkts=8,
#                 orig_pkts=9, fin_count=29, rst_count=30

CTU_RULES = [
    # rule 0: ddos is a volumetric flood - high rate is the single discriminating feature.
    # ctu ddos has median rate=645k. single condition avoids and-conjunction dead-rule failure.
    RuleTemplate(
        name="ddos_high_rate",
        feature_indices=[10],            # Rate
        condition_types=['gt'],
        init_thresholds=[1000.0],        # Rate > 1000 pps
    ),
    # rule 1: c&c-heartbeat - beaconing band (iat and rate both moderate and periodic)
    RuleTemplate(
        name="cc_heartbeat_beaconing",
        feature_indices=[6, 6, 10, 10],  # IAT_lo, IAT_hi, Rate_lo, Rate_hi
        condition_types=['gt', 'lt', 'gt', 'lt'],
        init_thresholds=[1.5, 5.0, 0.1, 5.0],  # IAT in (1.5,5), Rate in (0.1,5)
    ),
    # rule 2: benign / okiru single-packet probe - few packets, low rate
    RuleTemplate(
        name="benign_single_probe",
        feature_indices=[9, 10],         # orig_pkts, Rate
        condition_types=['lt', 'lt'],
        init_thresholds=[2.0, 100.0],    # orig_pkts<2, Rate<100
    ),
    # rule 3: okiru low-rate iot scan - low rate, short duration
    RuleTemplate(
        name="okiru_low_rate",
        feature_indices=[10, 23],        # Rate, duration
        condition_types=['lt', 'lt'],
        init_thresholds=[500.0, 2.5],    # Rate<500, duration<2.5
    ),
    # rule 4: asymmetric flood - originator sends, no reply (ddos / scan)
    RuleTemplate(
        name="asymmetric_flood",
        feature_indices=[8, 9],          # resp_pkts, orig_pkts
        condition_types=['lt', 'gt'],
        init_thresholds=[1.0, 1.0],      # resp_pkts<1, orig_pkts>1
    ),
    # rule 5: persistent connection - long-lived (c&c-hb, benign sessions)
    RuleTemplate(
        name="persistent_connection",
        feature_indices=[23],            # duration
        condition_types=['gt'],
        init_thresholds=[2.5],           # duration>2.5
    ),
    # rule 6: incomplete handshake - syn sent, no fin (c&c-hb, scan)
    RuleTemplate(
        name="incomplete_handshake",
        feature_indices=[21, 29],        # syn_count, fin_count
        condition_types=['gt', 'lt'],
        init_thresholds=[2.0, 1.0],      # syn_count>2, fin_count<1
    ),
    # rule 7: ddos short burst - high rate AND near-zero duration (volumetric flood)
    # replaces dead `high_variance_rate`. ddos has rate p50=645k AND duration~=0.
    RuleTemplate(
        name="ddos_short_burst",
        feature_indices=[10, 23],        # Rate, duration
        condition_types=['gt', 'lt'],
        init_thresholds=[1000.0, 0.1],   # Rate>1000, duration<0.1
    ),
]


# cic-iot-2023 rule templates
# all features z-scored. features (idx):
# rate=3, syn_count=12, fin_count=13, rst_count=14, tcp=22, udp=23,
# avg=33, iat=36, variance=38

CIC_RULES = [
    RuleTemplate(
        name="syn_flood_ddos",
        feature_indices=[3, 12],         # rate, syn_count
        condition_types=['gt', 'gt'],
        init_thresholds=[0.15, 1.5],     # rate z>0.15, syn_count z>1.5
    ),
    RuleTemplate(
        name="udp_mirai_flood",
        feature_indices=[22, 3],         # tcp, rate
        condition_types=['lt', 'gt'],
        init_thresholds=[-1.0, 0.0],     # tcp z<-1 (UDP-dominant), rate z>0
    ),
    RuleTemplate(
        name="recon_port_scan",
        feature_indices=[14],            # rst_count
        condition_types=['gt'],
        init_thresholds=[1.5],           # rst_count z>1.5
    ),
    RuleTemplate(
        name="vuln_scan",
        feature_indices=[13, 3],         # fin_count, rate
        condition_types=['gt', 'gt'],
        init_thresholds=[1.0, 0.0],      # fin_count z>1, rate z>0
    ),
    RuleTemplate(
        name="benign_low_rate",
        feature_indices=[3, 12],         # rate, syn_count
        condition_types=['lt', 'lt'],
        init_thresholds=[0.0, 0.0],      # rate z<0, syn_count z<0
    ),
    RuleTemplate(
        name="large_payload_flood",
        feature_indices=[33, 23],        # avg, udp
        condition_types=['gt', 'gt'],
        init_thresholds=[0.4, 0.0],      # avg z>0.4, udp z>0
    ),
    RuleTemplate(
        name="high_entropy_scan",
        feature_indices=[38, 14],        # variance, rst_count
        condition_types=['gt', 'gt'],
        init_thresholds=[0.5, 0.5],      # variance z>0.5, rst_count z>0.5
    ),
    RuleTemplate(
        name="short_connection_flood",
        feature_indices=[36, 3],         # iat, rate
        condition_types=['lt', 'gt'],
        init_thresholds=[-0.003, 0.1],   # iat z<-0.003 (near min), rate z>0.1
    ),
]


# one differentiable logical rule.
# score in (0, 1) - how well the sample matches the rule's conditions.
class DiffRule(nn.Module):
    def __init__(self, template: RuleTemplate):
        super().__init__()
        self.name = template.name
        self.feature_indices = template.feature_indices
        self.condition_types = template.condition_types
        n_cond = len(template.feature_indices)
        self.thresholds = nn.Parameter(
            torch.tensor(template.init_thresholds, dtype=torch.float32)
        )
        assert len(template.condition_types) == n_cond

    # x: (N, n_features)
    # k: steepness (higher = harder gates, approaches step function)
    # use_ste: straight-through estimator - binary gate in forward pass,
    # sigmoid gradient in backward pass. gives exact 0/1 outputs
    # at inference without quantization loss in training.
    # returns: (N,) rule activation scores in (0, 1)
    def forward(self, x: torch.Tensor, k: float = 1.0,
                use_ste: bool = False) -> torch.Tensor:
        score = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        for i, (feat_idx, ctype) in enumerate(
            zip(self.feature_indices, self.condition_types)
        ):
            xi = x[:, feat_idx]
            theta = self.thresholds[i]
            if ctype == 'gt':
                gate = torch.sigmoid(k * (xi - theta))
            else:  # 'lt'
                gate = torch.sigmoid(k * (theta - xi))
            if use_ste:
                # ste: forward uses hard binary step, backward uses sigmoid gradient
                gate = gate + ((gate > 0.5).float() - gate).detach()
            score = score * gate
        return score  # (N,)

    # fraction of activations with |gate - 0.5| > 0.4 (near 0 or near 1).
    # high crispness = rule is making decisive predictions.
    def get_crispness(self, x: torch.Tensor, k: float) -> float:
        with torch.no_grad():
            crisp_fracs = []
            for i, (feat_idx, ctype) in enumerate(
                zip(self.feature_indices, self.condition_types)
            ):
                xi = x[:, feat_idx]
                theta = self.thresholds[i]
                if ctype == 'gt':
                    gate = torch.sigmoid(k * (xi - theta))
                else:
                    gate = torch.sigmoid(k * (theta - xi))
                crisp = (torch.abs(gate - 0.5) > 0.4).float().mean().item()
                crisp_fracs.append(crisp)
        return float(torch.tensor(crisp_fracs).mean())


# M differentiable rules -> class logits.
# rule_scores (N, M) -> W (M, n_classes) -> logits (N, n_classes)
class RuleBank(nn.Module):
    def __init__(self, templates: List[RuleTemplate], n_classes: int):
        super().__init__()
        self.rules = nn.ModuleList([DiffRule(t) for t in templates])
        self.n_rules = len(templates)
        self.n_classes = n_classes
        # weight matrix: maps rule activations to class logits
        self.W = nn.Linear(self.n_rules, n_classes, bias=True)
        # rule importance: learned per-rule weight, used for sparsity loss
        self.rule_importance = nn.Parameter(torch.ones(self.n_rules))

    # returns logits (N, n_classes) and rule_scores (N, M).
    def forward(self, x: torch.Tensor, k: float = 1.0, use_ste: bool = False):
        scores = torch.stack(
            [rule(x, k, use_ste=use_ste) for rule in self.rules], dim=1
        )  # (N, M)
        # weight by learned importance before classification
        weighted = scores * torch.sigmoid(self.rule_importance).unsqueeze(0)
        logits = self.W(weighted)  # (N, n_classes)
        return logits, scores  # return raw scores for OOD

    # hard rule activations at high k with ste (exact binary at eval).
    def get_rule_activations(self, x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack(
                [rule(x, k, use_ste=True) for rule in self.rules], dim=1
            )  # (N, M)


# small mlp for patterns not covered by rules.
class NeuralFallback(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# neuro-symbolic nids model.
# final_logits = alpha * rule_logits + (1-alpha) * neural_logits
# where alpha = sigmoid(gate_weight) is learned globally.
# ood scoring: max_j rule_score_j (rule confidence).
# high = known pattern; low = unseen = likely ood.
class NeSyNIDS(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        rule_templates: List[RuleTemplate],
        neural_hidden: int = 128,
        global_gate: bool = True,
    ):
        super().__init__()
        self.rule_bank = RuleBank(rule_templates, n_classes)
        self.neural_fallback = NeuralFallback(in_dim, n_classes, neural_hidden)
        self.in_dim = in_dim

        # gating: global scalar alpha in (0,1) or per-sample via small network
        if global_gate:
            # learned global balance between rules and neural
            self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
            self.global_gate = True
        else:
            # per-sample gate: small network predicts alpha from input
            self.gate_net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            self.global_gate = False

    # returns:
    # logits (N, n_classes) - combined prediction
    # rule_scores (N, M)    - raw rule activations for ood + analysis
    def forward(self, x: torch.Tensor, k: float = 1.0, use_ste: bool = False):
        rule_logits, rule_scores = self.rule_bank(x, k, use_ste=use_ste)
        neural_logits = self.neural_fallback(x)

        if self.global_gate:
            alpha = torch.sigmoid(self.gate)
        else:
            alpha = torch.sigmoid(self.gate_net(x))  # (N, 1)

        logits = alpha * rule_logits + (1.0 - alpha) * neural_logits
        return logits, rule_scores

    # ood score = max rule activation at hard k.
    # higher = more confidence = less likely ood.
    # negate for auroc (higher score -> more ood).
    def get_ood_score(self, x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
        with torch.no_grad():
            scores = self.rule_bank.get_rule_activations(x, k)  # (N, M)
            max_rule_conf = scores.max(dim=1).values             # (N,)
        return -max_rule_conf  # flip: high = OOD

    # rule activation vector as interpretable embedding for ood.
    def get_embedding(self, x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
        with torch.no_grad():
            return self.rule_bank.get_rule_activations(x, k)

    # compute per-rule statistics:
    # - mean activation per known class
    # - crispness at hard k
    # - learned threshold values
    def get_rule_summary(self, x: torch.Tensor, y: torch.Tensor, k: float = 10.0,
                         id_to_label: dict = None) -> dict:
        self.eval()
        with torch.no_grad():
            scores = self.rule_bank.get_rule_activations(x, k)  # (N, M)
        n_rules = len(self.rule_bank.rules)
        n_classes = len(torch.unique(y))
        result = {}
        for r_idx, rule in enumerate(self.rule_bank.rules):
            rule_scores_r = scores[:, r_idx].cpu().numpy()
            class_means = {}
            for cls in torch.unique(y).tolist():
                mask = (y == cls).cpu().numpy()
                class_means[cls] = float(rule_scores_r[mask].mean()) if mask.any() else 0.0
            crispness = rule.get_crispness(x, k)
            thresholds = rule.thresholds.detach().cpu().tolist()
            result[rule.name] = {
                "class_means": class_means,
                "crispness": crispness,
                "thresholds": thresholds,
                "conditions": list(zip(rule.feature_indices, rule.condition_types)),
            }
        return result

    # returns current alpha (rule weight in [0,1]).
    def get_gate_value(self) -> float:
        if self.global_gate:
            return float(torch.sigmoid(self.gate).item())
        return float('nan')  # per-sample gate


# ablation: rules only, no neural fallback.
class RuleOnlyNIDS(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, rule_templates: List[RuleTemplate]):
        super().__init__()
        self.rule_bank = RuleBank(rule_templates, n_classes)

    def forward(self, x: torch.Tensor, k: float = 1.0, use_ste: bool = False):
        return self.rule_bank(x, k, use_ste=use_ste)  # (logits, rule_scores)

    def get_ood_score(self, x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
        with torch.no_grad():
            scores = self.rule_bank.get_rule_activations(x, k)
        return -scores.max(dim=1).values

    def get_embedding(self, x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
        with torch.no_grad():
            return self.rule_bank.get_rule_activations(x, k)
