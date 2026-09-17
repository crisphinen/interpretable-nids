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
0.752→0.724) — an honest nuance now documented.

**C3 — Mahalanobis on binary vectors not justified vs. alternatives.**
Added a scorer ablation (new Table, 5 seeds) comparing per-class Mahalanobis
with two binary-simplex distances (Hamming and Jaccard to the nearest known-class
majority-vote centroid), a logit-space energy score, and a Mahalanobis+energy
combination, all under one protocol (class statistics fitted on training
activations, AUROC on known-test vs unknown-test). On CTU every activation-space
scorer works (Mahalanobis 0.899, Jaccard 0.891, Hamming 0.863); on CIC none
exceeds 0.64 (Mahalanobis 0.615) and only the logit-space energy score restores
detection (0.749). The representation, not the choice among activation-space
metrics, decides whether OOD scoring works; this is now a headline contribution.
(While reconstructing this ablation we also aligned the NeSy-NIDS evaluation with
the CBM one — per-class Mahalanobis fitted on training activations and scored on
the known test split — which is the protocol the paper describes, and retrained
all models from the released scripts on the released data; every number in the
paper now comes from those scripts.)

**C4 — CBM single-seed vs. NeSy 5-seed.**
Fixed. All learned models (MLP, JointCBM ×4 γ, SequentialCBM, HybridCBM,
NeSy-NIDS) now report 5-seed mean±std in Table 1. This directly surfaced the
artifact: the single-seed γ differences (CTU 0.838/0.895/0.883/0.887 and CIC
0.668/0.654/0.727/0.707 for γ=0/0.1/0.5/1.0) do not replicate — over five seeds
the CTU values are 0.800/0.828/0.835/0.840 and the CIC values 0.746/0.739/0.727/
0.720 (see C-central below). SequentialCBM's prior CTU 0.707 was an unlucky seed
(5-seed mean 0.885).

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
unknown, C&C-HeartBeat demoted to unknown) and re-ran the CBM evaluation (5
seeds). The conclusions hold: every JointCBM variant is within 0.005 of the MLP
on F1 (0.994), and γ does not buy OOD detection (MLP 0.841; JointCBM γ=0: 0.721,
γ=1.0: 0.667 — a decrease of ~1.5 seed-std, in the opposite direction to the
ordering on the main CTU split and under the temporal split of R2-C3). We repeat
the same check on CIC-IoT-2023 (known set {Benign, DDoS-ICMP_Flood,
Mirai-udpplain, Recon-PortScan, VulnerabilityScan}, 5 seeds) with the same
outcome: F1 within 0.002 of the MLP (0.819) and a mild decrease with γ as on the
main CIC split (γ=0: 0.752, γ=0.1: 0.740, γ=0.5: 0.740, γ=1.0: 0.727). Both
datasets thus confirm the findings are not artifacts of the original partition.

---

**Central revision (arising from C4/C8) — the task-coupling penalty does not replicate.**
The originally reported CIC penalty (JointCBM AUROC rising to a γ=0.5 peak of
0.727) rested on one seed whose γ=0 draw was anomalously low (0.668; 5-seed mean
0.746). Across five seeds there is no peak: AUROC decreases monotonically with γ
(0.746/0.739/0.727/0.720). An independent second five-seed set (seeds 5–9) gives
the same picture with a shallower slope (γ=0: 0.749, γ=0.5: 0.743, γ=1.0: 0.741;
pooled over ten seeds γ=0 0.747±0.014 vs γ=0.5 0.735±0.013). On CTU the ordering is the reverse
(0.800→0.840), and on the alternative CTU split it reverses again; across four
dataset/split combinations the sign of the γ effect flips twice and its size
never exceeds 0.04. We have removed the task-coupling claim and the γ-vs-AUROC
figure, retitled and re-abstracted the paper, and now present concept accuracy
as the property γ controls, decoupled from OOD. We report the non-replication
itself as a methodological result: single-seed ablations can manufacture
spurious open-set effects.
