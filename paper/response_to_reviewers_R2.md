# Response to Reviewer 2

We thank the reviewer for the detailed review. In response we added five
experiments (stronger OOD baselines, a concept-leakage probe, a full α-gate
sweep, a temporal/session-respecting split, and single-thread batch-1 latency),
quantified the two figure claims, and restated the contribution more precisely.
Point-by-point:

---

**C1 — What is fundamentally new vs. application-specific integration?**
We agree and now state this explicitly (new "Scope of the contribution" paragraph,
§I). NeSy-NIDS is a purpose-built *integration* of established components; we claim
no new learning algorithm. The novelty is empirical/methodological: a controlled,
multi-seed open-set study that (i) identifies the scorer–representation coupling
that governs interpretable-model OOD, and (ii) overturns a plausible but
seed-fragile "task-coupling" claim. We frame establishing which mechanisms
transfer to open-set NIDS—under replication and stronger baselines—as the value.

**C2 — CBM single seed.**
Fixed. All learned models (MLP, JointCBM ×4 γ, SequentialCBM, HybridCBM,
NeSy-NIDS) now report 5-seed mean±std (Table I). This directly surfaced that the
single-seed CTU γ differences (0.838/0.895/0.883/0.887 for γ=0/0.1/0.5/1.0) are
within seed variance (5-seed: 0.840/0.837/0.840/0.850, std 0.025–0.053); see
C-central.

**C3 — No separation by device/capture session.**
Added a temporal, session-respecting evaluation (§IV). CTU attacks originate from
very few source hosts (Okiru 4, DDoS 7, C&C-HB 5 src_ips), so a clean
device-disjoint attack split is infeasible; we instead split each known class by
capture timestamp (earliest-70% train / latest-15% test; 3 seeds). Under this
harder protocol F1 drops ≈0.06 (MLP 0.933→0.871)—confirming the random split was
mildly optimistic—but the core claims hold: every JointCBM variant still matches
the MLP on F1 (0.870–0.872 vs 0.871), and OOD AUROC brackets the MLP's 0.801
(0.703/0.773/0.807/0.836 for γ=0/0.1/0.5/1.0, seed-std up to 0.08). The γ
ordering here is the reverse of the one on the alternative CTU split (R1-C8),
further evidence that γ has no consistent OOD effect.

**C4 — OOD baselines too limited.**
Added Table (post-hoc detectors on the MLP): MSP, ODIN, energy, kNN, Mahalanobis.
Confidence scores are weak (CTU 0.46–0.48); feature-space detectors set the
ceiling (Mahalanobis 0.890 CTU, kNN 0.842 CIC; single seed). We now state explicitly that we
make **no SOTA open-set claim**—the contribution is competitive OOD *with
auditability*, plus the scorer–representation finding.

**C5 — Mahalanobis on binary vectors under-validated.**
Addressed by the scorer ablation (Table, 5 seeds), which compares per-class
Mahalanobis against Hamming and Jaccard distances to per-class binary centroids,
an energy score and a combination, all under one protocol. Where activations are
well separated (CTU) every activation-space scorer works and Mahalanobis is best
(0.911 vs 0.880/0.847); where they are near-deterministic (CIC) no activation-space
scorer exceeds 0.64 and the logit-space energy score is needed (0.728). We state in
the text that Mahalanobis is an approximation on binary vectors and that the scorer
must be chosen to fit the representation.

**C6 — "γ eliminates concept leakage" not demonstrated.**
We replaced the concept-accuracy proxy with a direct leakage probe (linear probe
from soft vs. binarised concepts to class; leakage = accuracy gap). Leakage is
negligible on CTU (≤0.002, all γ) and flat at 0.05–0.08 on CIC with no reduction
as γ grows (γ=0: 0.067, γ=0.5: 0.075). We have **retracted** the "γ eliminates leakage" claim.

**C7 — α-Pareto unsupported (only two λα points).**
Added a full λα sweep over six values × 5 seeds (Table). λα monotonically tunes α
from ~0.30–0.37 to ~0.86–0.95 at zero F1 cost and with a small OOD side effect
(AUROC −0.02 on CTU, +0.04 on CIC across the full range, about one to two
seed-std); we now describe it as a control knob rather than a steep Pareto front. Each sweep point was trained
from scratch; we no longer claim that λα can be set by fine-tuning without
retraining, and state this as untested.

**C8 — 100% crispness is trivial.**
Agreed. In revisiting this we also found that the method description was
imprecise: the hard 0.5 threshold is applied at inference (for the audited rule
vector and OOD scoring), while training uses the annealed sigmoid gates
throughout; the straight-through gradient is not part of the training path. We
have corrected §III-C2 and Alg. 2 to say exactly this and no longer describe
binarity as a result. All reported numbers were produced by this implementation,
so none change on this account. We replaced the trivial post-threshold crispness
with a meaningful measure — the fraction of soft activations at k=10 within 0.1
of binary: 0.86 (CTU), 0.77 (CIC) — and verify the thresholding is benign for
predictions (class changes on 0.0% of CTU and 1.1% of CIC validation flows).

**C9 — Compute analysis too optimistic.**
Corrected (§Computational Overhead). We report footprint (<0.4 MB; NeSy 13,825
params) and, crucially, batch-1 single-thread latency (MLP 96 µs, JointCBM 109 µs,
NeSy ~580 µs) rather than the batch-512 0.42 µs figure. We no longer claim measured
IoT-hardware performance; on-device ARM validation is future work.

**C10 — Fig 3/4 claims unsupported by ablation/correlation.**
Quantified both. Fig 3: across all 10 NeSy models, mean within-class activation
variance correlates with OOD AUROC at **Pearson r=0.976 (p<10⁻⁵)**; we note in
the text that the two datasets form two clusters, so this confirms the
between-dataset pattern rather than a fine-grained law. Fig 4: drift
reproducibility is strong on CIC (mean pairwise r=0.71) but seed-dependent on CTU
(r=0.04); we now report this honestly and no longer claim uniform semantic
consistency.

---

**Central revision (shared with Reviewer 1).** The 5-seed evaluation the reviewers
requested revealed that our "task-coupling OOD penalty" was a single-seed artifact
(the CIC γ=0.5 peak does not survive replication and reverses on an independent seed
set). We retitled and reframed the paper around what the multi-seed evidence
supports—representation geometry and scorer choice, not regularization strength,
govern interpretable open-set detection.
