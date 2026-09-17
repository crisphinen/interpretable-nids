# Response to Reviewer 1

We thank the reviewer for the careful review. Following the concern about
statistical rigor (Comments 4 and 8), we re-ran every learned model over five
seeds. This showed that the originally reported "task-coupling OOD penalty" was a
single-seed artifact that vanishes, and reverses, under replication. We have
therefore **reframed the paper** around what the multi-seed evidence supports:
representation geometry and scorer choice, not concept-regularization strength,
govern open-set detectability. The paper is retitled "Interpretable Open-Set
Intrusion Detection for IoT: What Governs Out-of-Distribution Detection in
Concept-Bottleneck and Neuro-Symbolic Models." Point-by-point responses follow.

---

**C1 — Known/unknown selection not justified.**
Added a "Known/unknown selection" paragraph (Sec. Datasets): the known set spans
the coarse traffic archetypes an operator can realistically label early in
deployment (benign + one representative per dominant malicious category —
volumetric DDoS, botnet C&C/Mirai, reconnaissance), with held-out unknowns being
predominantly sub-families/variants of those categories, i.e., the realistic
novel-variant open-set regime. See also C8.

**C2 — λ grid underspecified / limited exploration.**
Clarified (Sec. Setup) that λ weights *concept supervision* and was selected by
grid search over {0.1,0.5,1.0} on the criterion of known-class F1, which is
invariant to λ (±0.002). λ was therefore fixed at the mid-value and γ studied as
the variable of interest. New Table (λ-sensitivity, 5 seeds) reports F1, concept
accuracy, and OOD AUROC across λ: F1 is flat; λ trades concept accuracy against
OOD in dataset-dependent directions (λ=0.1→1.0: CTU AUROC 0.790→0.884, CIC
0.753→0.681, both beyond one seed-std) — an honest nuance now documented.

**C3 — Mahalanobis on binary vectors not justified vs. alternatives.**
Added a scorer ablation (new Table, 5 seeds) comparing per-class Mahalanobis
with two binary-simplex distances (Hamming and Jaccard to the nearest known-class
majority-vote centroid), a logit-space energy score, and a Mahalanobis+energy
combination, all under one protocol (class statistics fitted on training
activations, AUROC on known-test vs unknown-test). On CTU every activation-space
scorer works and the covariance-aware Mahalanobis is best (0.911 vs Jaccard 0.880,
Hamming 0.847); on CIC no activation-space scorer exceeds 0.64 (Mahalanobis 0.631)
and the energy score is needed (0.728). The scorer therefore has to be chosen to
fit the representation geometry; this is now a headline contribution. (While
reconstructing this ablation we also aligned the NeSy-NIDS evaluation with the
CBM one — per-class Mahalanobis fitted on training activations and scored on the
known test split — which is the protocol the paper describes; all NeSy OOD
numbers were regenerated under it.)

**C4 — CBM single-seed vs. NeSy 5-seed.**
Fixed. All learned models (MLP, JointCBM ×4 γ, SequentialCBM, HybridCBM,
NeSy-NIDS) now report 5-seed mean±std in Table 1. This directly surfaced the
artifact: the single-seed γ differences (CTU 0.838/0.895/0.883/0.887 and CIC
0.668/0.654/0.727/0.707 for γ=0/0.1/0.5/1.0) are within seed variance and do not
replicate (see C-central below). SequentialCBM's prior CTU 0.707 was an unlucky
seed (5-seed mean 0.898).

**C5 — "Joint evaluation" novelty weak.**
Reframed around a *unified open-set evaluation framework*: two structurally
distinct interpretable paradigms held to one protocol (identical splits, scoring,
thresholding, seeds). The two findings that survive — scorer–representation
matching and the multi-seed refutation of a regularization effect — are only
visible under this controlled design (abstract, contributions, discussion).

**C6 — Exact concept definitions/thresholds.**
Added a full concept-definition table (both datasets, all 8+8 concepts with exact
thresholds; CTU raw units, CIC standardized units).

**C7 — Why exclude TON_IoT.**
Strengthened rationale (Sec. Datasets): TON_IoT's features are not the Zeek-style
flow statistics our concept/rule vocabularies are defined over, so it would
require re-deriving the entire vocabulary and confound a same-interface
comparison; its taxonomy is also shallower (~9 vs 13/34 classes). Regarding whether the
"task-coupling OOD penalty" would generalize: that claim has been withdrawn (see
Central revision below), as it did not survive multi-seed replication, so the
question is moot. The effects that do survive are properties of the training
objective and representation rather than of a specific dataset, and we expect
them to transfer.

**C8 — Sensitivity to the known/unknown split.**
Added an alternative CTU split ({Benign, DDoS, Okiru, C&C}; C&C promoted from
unknown, C&C-HeartBeat demoted to unknown) and re-ran the CBM evaluation (3
seeds). The conclusions hold: interpretable models match the MLP on F1 (within
0.001 of 0.930), and γ does not buy OOD detection (MLP 0.899; JointCBM γ=0:
0.840, γ=1.0: 0.797 — a decrease within ~1.4 seed-std, and in the opposite
direction to the ordering under the temporal split of R2-C3). We repeat the same
check on CIC-IoT-2023 (known set {Benign, DDoS-ICMP_Flood, Mirai-udpplain,
Recon-PortScan, VulnerabilityScan}, 3 seeds) with the same outcome: F1 within
0.002 of the MLP (0.811) and no monotone γ ordering (γ=0: 0.876, γ=0.1: 0.813,
γ=0.5: 0.849, γ=1.0: 0.784). Both datasets thus confirm the findings are not
artifacts of the original partition.

---

**Central revision (arising from C4/C8) — the task-coupling penalty does not replicate.**
The originally reported CIC penalty (JointCBM AUROC rising to a γ=0.5 peak of
0.727) rested on one seed whose γ=0 draw was anomalously low (0.668; 5-seed mean
0.718). Across five seeds, γ∈{0,0.1,0.5} form a flat plateau (0.717–0.726) within
std. An independent second five-seed set (seeds 5–9) reverses the direction
(γ=0: 0.735, γ=0.5: 0.702). Pooled over ten seeds the variants are statistically
indistinguishable (γ=0.5: 0.714±0.028 vs γ=0: 0.726±0.031). We have removed the
task-coupling claim and the γ-vs-AUROC figure, retitled and re-abstracted the
paper, and now present concept-fidelity (leakage) reduction as a property of γ in
its own right, decoupled from OOD. We report the non-replication itself as a
methodological result: single-seed ablations can manufacture spurious open-set
effects.
