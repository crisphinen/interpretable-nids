#!/usr/bin/env python3
# NeSy-NIDS fidelity (tab:fidelity NeSy rows, tab:headagree, tab:alpha, tab:rulequality), on the
# known test split at the evaluation steepness K_EVAL, from the alpha-sweep checkpoints:
#   alpha         learned global gate sigmoid(gate)
#   auroc         per-class Mahalanobis OOD AUROC (from the per-seed eval JSON)
#   dep_neural    deployed prediction == neural-fallback-only prediction
#   dep_rule      deployed prediction == rule-head-only prediction
#   hard_soft     hard (STE) rule conjunction == soft rule product thresholded at 0.5
#   crisp         soft activations within 0.1 of 0 or 1
#   hard_pred     deployed prediction changes when classifying from the hard (STE) activations
# Rule quality uses the lambda_alpha=0 models: coverage (hard activation), dominant class and its
# precision among activating flows, precision for the declared target class (tab:rulebank), and
# the relative std of the learned thresholds across seeds (mean std over mean |threshold|).
# Writes results/nesy/fidelity_{ds}.json.
#
# usage: python -m nesy.run_fidelity --datasets ctuk cicr --device cuda
import argparse, json

import numpy as np
import torch

from nesy.evaluate import load_known_split, load_model, K_EVAL, RESULTS_DIR

LAMBDAS = ["0", "0.05", "0.1", "0.2", "0.5", "1.0"]
SEEDS = range(5)
TARGETS = {  # declared target class per rule (tab:rulebank / tab:rulequality)
    "ddos_high_rate": "DDoS", "cc_heartbeat_beaconing": "C&C-HeartBeat", "benign_single_probe": "Benign",
    "okiru_low_rate": "Okiru", "asymmetric_flood": "DDoS", "persistent_connection": "C&C-HeartBeat",
    "incomplete_handshake": "C&C-HeartBeat", "ddos_short_burst": "DDoS",
    "syn_flood_ddos": "DDoS-SYN_Flood", "udp_mirai_flood": "Mirai-greeth_flood",
    "recon_port_scan": "Recon-PortScan", "vuln_scan": "VulnerabilityScan", "benign_low_rate": "Benign_Final",
    "large_payload_flood": "Benign_Final", "high_entropy_scan": "Recon-PortScan",
    "short_connection_flood": "DDoS-SYN_Flood",
}


def ckpt(ds, s, la):
    tag = "" if la == "0" else f"_a{la}"
    return RESULTS_DIR / f"{ds}_nesy_s{s}{tag}.pt", RESULTS_DIR / f"{ds}_nesy_s{s}{tag}_eval.json"


@torch.no_grad()
def branch_stats(model, X):
    logits, soft = model(X, k=K_EVAL)
    dep = logits.argmax(1)
    neural = model.neural_fallback(X).argmax(1)
    rule_logits, _ = model.rule_bank(X, K_EVAL)
    hard = model.rule_bank.get_rule_activations(X, K_EVAL)
    return {"alpha": float(torch.sigmoid(model.gate).item()),
            "dep_neural": float((dep == neural).float().mean()),
            "dep_rule": float((dep == rule_logits.argmax(1)).float().mean()),
            "hard_soft": float(((soft > 0.5).float() == hard).float().mean()),
            "crisp": float(((soft < 0.1) | (soft > 0.9)).float().mean()),
            "hard_pred": float((model(X, k=K_EVAL, use_ste=True)[0].argmax(1) != dep).float().mean())}


@torch.no_grad()
def rule_quality(ds, X, y, id_to_label, device):
    names, acts, thetas = None, [], []
    for s in SEEDS:
        m, _ = load_model(ckpt(ds, s, "0")[0], device)
        names = [r.name for r in m.rule_bank.rules]
        acts.append(m.rule_bank.get_rule_activations(X, K_EVAL).cpu().numpy() > 0.5)
        thetas.append([r.thresholds.detach().cpu().numpy() for r in m.rule_bank.rules])
    y = y.cpu().numpy()
    out = {}
    for j, name in enumerate(names):
        per = []
        for a in acts:
            on = a[:, j]
            counts = np.bincount(y[on], minlength=len(id_to_label)) if on.any() else np.zeros(len(id_to_label))
            dom = int(np.argmax(counts)); tot = max(on.sum(), 1)
            tgt = [i for i, l in id_to_label.items() if l == TARGETS[name]][0]
            per.append({"cov": float(on.mean()), "dom": id_to_label[dom], "prec": float(counts[dom] / tot),
                        "tprec": float(counts[tgt] / tot)})
        th = np.stack([t[j] for t in thetas])                      # seeds x conditions
        # mean across-seed std over mean |threshold| (ratio of averages)
        rel = th.std(0).mean() / max(np.abs(th.mean(0)).mean(), 1e-9)
        doms = [p["dom"] for p in per]
        out[name] = {"coverage": float(np.mean([p["cov"] for p in per])),
                     "dominant": max(set(doms), key=doms.count),
                     "precision": float(np.mean([p["prec"] for p in per])),
                     "target": TARGETS[name],
                     "target_precision": float(np.mean([p["tprec"] for p in per])),
                     "target_precision_std": float(np.std([p["tprec"] for p in per])),
                     "threshold_rel_std": float(rel)}
    return out


def threshold_drift(ds, device):
    """Per-seed drift vectors (learned - initial, every rule condition) of the lambda_alpha=0
    models and their mean pairwise Pearson r across seeds (fig:threshdrift)."""
    from nesy.model import rules_for
    templates = rules_for(ds)
    labels = [f"{t.name}[c{j + 1}:{'>' if c == 'gt' else '<'}]"
              for t in templates for j, c in enumerate(t.condition_types)]
    vecs = []
    for s in SEEDS:
        m, _ = load_model(ckpt(ds, s, "0")[0], device)
        vecs.append([float(l - i) for r, t in zip(m.rule_bank.rules, templates)
                     for l, i in zip(r.thresholds.detach().cpu().tolist(), t.init_thresholds)])
    V = np.array(vecs)
    rs = [np.corrcoef(V[a], V[b])[0, 1] for a in range(len(V)) for b in range(a + 1, len(V))]
    return {"labels": labels, "drift_per_seed": vecs, "mean_pairwise_r": float(np.mean(rs))}


def run(ds, device):
    X, y, id_to_label = load_known_split(ds, device, "test_known")
    sweep = {}
    for la in LAMBDAS:
        rows = []
        for s in SEEDS:
            ck, ev = ckpt(ds, s, la)
            if not ck.exists():
                continue
            m, _ = load_model(ck, device)
            r = {"seed": s, **branch_stats(m, X)}
            if ev.exists():
                r["auroc"] = json.loads(ev.read_text())["ood_auroc_mahal"]
            rows.append(r)
        if rows:
            sweep[la] = {"rows": rows, **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}}
            print(f"[{ds}] la={la:4s} n={len(rows)} " + " ".join(
                f"{k}={sweep[la][k]:.3f}" for k in ("alpha", "auroc", "dep_neural", "dep_rule", "hard_soft", "crisp")
                if k in sweep[la]))
    rq = rule_quality(ds, X, y, id_to_label, device)
    for n, v in rq.items():
        print(f"[{ds}] {n:24s} cov={v['coverage']:.2f} dom={v['dominant']:18s} prec={v['precision']:.2f} "
              f"target={v['target']:18s} tprec={v['target_precision']:.2f}({v['target_precision_std']:.2f}) "
              f"thr={v['threshold_rel_std']:.2f}")
    drift = threshold_drift(ds, device)
    print(f"[{ds}] threshold drift: mean pairwise r across seeds = {drift['mean_pairwise_r']:.2f}")
    (RESULTS_DIR / f"fidelity_{ds}.json").write_text(
        json.dumps({"sweep": sweep, "rule_quality": rq, "threshold_drift": drift}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for ds in args.datasets:
        run(ds, device)


if __name__ == "__main__":
    main()
