#!/usr/bin/env python3
# Rule quality at the expert-written INITIAL thresholds, for comparison with the learned
# thresholds of nesy/run_fidelity.py (tab:rulequality). Hard rule activations depend only on the
# thresholds, so a seed-0 lambda_alpha=0 model is loaded and its thresholds are reset to the
# template values. Same known test split and statistics as run_fidelity.rule_quality: coverage,
# dominant class and its precision, and precision for the declared target class.
# Writes results/nesy/rule_init_{ds}.json.
#
# usage: python -m nesy.run_rule_init --datasets ctuk cicr --device cuda
import argparse, json

import numpy as np
import torch

from nesy.evaluate import load_known_split, load_model, K_EVAL, RESULTS_DIR
from nesy.model import rules_for
from nesy.run_fidelity import TARGETS, ckpt


@torch.no_grad()
def run(ds, device):
    X, y, id_to_label = load_known_split(ds, device, "test_known")
    m, _ = load_model(ckpt(ds, 0, "0")[0], device)
    for r, t in zip(m.rule_bank.rules, rules_for(ds)):
        r.thresholds.copy_(torch.tensor(t.init_thresholds, dtype=r.thresholds.dtype, device=device))
    acts = m.rule_bank.get_rule_activations(X, K_EVAL).cpu().numpy() > 0.5
    y = y.cpu().numpy(); out = {}
    learned = json.loads((RESULTS_DIR / f"fidelity_{ds}.json").read_text())["rule_quality"]
    for j, rule in enumerate(m.rule_bank.rules):
        on = acts[:, j]
        counts = np.bincount(y[on], minlength=len(id_to_label)) if on.any() else np.zeros(len(id_to_label))
        tot = max(on.sum(), 1); dom = int(np.argmax(counts))
        tgt = [i for i, l in id_to_label.items() if l == TARGETS[rule.name]][0]
        out[rule.name] = {"coverage": float(on.mean()), "dominant": id_to_label[dom],
                          "precision": float(counts[dom] / tot), "target": TARGETS[rule.name],
                          "target_precision": float(counts[tgt] / tot)}
        L = learned[rule.name]
        print(f"[{ds}] {rule.name:24s} init cov={out[rule.name]['coverage']:.2f} tprec={out[rule.name]['target_precision']:.2f} "
              f"dom={out[rule.name]['dominant'][:14]:14s} | learned cov={L['coverage']:.2f} tprec={L['target_precision']:.2f} "
              f"dom={L['dominant'][:14]}")
    (RESULTS_DIR / f"rule_init_{ds}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for ds in args.datasets:
        run(ds, torch.device(args.device if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
