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
γ differences (0.890/0.895/0.883) are within seed variance; see C-central.

**C3 — No separation by device/capture session.**
Added a temporal, session-respecting evaluation (§IV). CTU attacks originate from
very few source hosts (Okiru 4, DDoS 7, C&C-HB 5 src_ips), so a clean
device-disjoint attack split is infeasible; we instead split each known class by
capture timestamp (earliest-70% train / latest-15% test). Under this harder
protocol F1 drops ≈0.06 (MLP 0.933→0.871)—confirming the random split was mildly
optimistic—but the core claims hold: interpretable models still match the MLP at
zero F1 cost (JointCBM γ=0.5: 0.872 vs MLP 0.871) and retain comparable OOD AUROC
(0.806 vs 0.801).

**C4 — OOD baselines too limited.**
Added Table (post-hoc detectors on the MLP): MSP, ODIN, energy, kNN, Mahalanobis.
Confidence scores are weak (CTU 0.46–0.48); feature-space detectors set the
ceiling (Mahalanobis 0.890 CTU, kNN 0.842 CIC). We now state explicitly that we
make **no SOTA open-set claim**—the contribution is competitive OOD *with
auditability*, plus the scorer–representation finding.

**C5 — Mahalanobis on binary vectors under-validated.**
Addressed by the scorer ablation (Table, 5 seeds): Hamming/Jaccard (native to the
binary simplex) are near-random (0.52–0.59); Mahalanobis is justified where
activations are separable (CTU 0.906), and a logit-space energy score recovers
detection where they are non-Gaussian (CIC 0.605→0.730).

**C6 — "γ eliminates concept leakage" not demonstrated.**
We replaced the concept-accuracy proxy with a direct leakage probe (linear probe
from soft vs. binarised concepts to class; leakage = accuracy gap). Leakage is
negligible on CTU (≤0.001, all γ) and flat at 0.06–0.07 on CIC with no reduction
as γ grows. We have **retracted** the "γ eliminates leakage" claim.

**C7 — α-Pareto unsupported (only two λα points).**
Added a full λα sweep over six values × 5 seeds (Table). λα monotonically tunes α
from ~0.30–0.37 to ~0.86–0.95 at zero F1 cost and near-zero OOD cost (AUROC
varies ≤0.02 on CTU, ≤0.04 on CIC, within one seed-std); we now describe it
as a tunable control knob rather than a steep Pareto front.

**C8 — 100% crispness is trivial.**
Agreed; softened. We now state crispness follows directly from the STE
hard-threshold operation; the non-trivial property is reaching it via k-annealing
while keeping thresholds learnable at zero accuracy cost.

**C9 — Compute analysis too optimistic.**
Corrected (§Computational Overhead). We report footprint (<0.4 MB; NeSy 13,825
params) and, crucially, batch-1 single-thread latency (MLP 96 µs, JointCBM 109 µs,
NeSy ~580 µs) rather than the batch-512 0.42 µs figure. We no longer claim measured
IoT-hardware performance; on-device ARM validation is future work.

**C10 — Fig 3/4 claims unsupported by ablation/correlation.**
Quantified both. Fig 3: across all 10 NeSy models, mean within-class activation
variance correlates with OOD AUROC at **Pearson r=0.986 (p<10⁻⁴)**. Fig 4: drift
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
