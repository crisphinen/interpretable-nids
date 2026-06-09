# concept engineering for iot nids - ctu-iot-23 and cic-iot-2023.
# each concept is a binary label derived from raw flow features.
# concepts are interpretable security behaviors a soc analyst can understand and override.
#
# ctu concepts (8):
# is_beaconing        - periodic low-rate c&c heartbeat
# is_high_rate        - volumetric flood (ddos)
# is_syn_heavy        - syn-dominant flow (flood or scan)
# is_short_connection - sub-millisecond or zero-duration connection
# is_persistent       - long-lived connection (> 2.5 s)
# is_asymmetric       - originator sends, no response (one-sided)
# is_single_packet_probe  - single packet, no sustained data exchange
# is_incomplete_handshake - syn without fin (flood / scan pattern)
#
# cic concepts (8):
# is_high_rate        - high packet rate
# is_syn_flood        - large syn count
# is_udp_dominant     - udp-heavy flood
# is_short_connection - near-zero duration
# is_large_payload    - large avg packet size
# is_port_scan        - rst-heavy (closed port responses)
# is_high_variance    - bursty traffic (high iat std)
# is_persistent       - long-lived connection

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ctu-iot-23 concept rules

CTU_CONCEPTS = [
    "is_beaconing",
    "is_high_rate",
    "is_syn_heavy",
    "is_short_connection",
    "is_persistent",
    "is_asymmetric",
    "is_single_packet_probe",
    "is_incomplete_handshake",
]


# returns (N, 8) float32 binary concept labels for ctu-iot-23.
# -1 sentinel values are treated as 0/absent for concept rules.
def ctu_concept_labels(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    C = np.zeros((n, len(CTU_CONCEPTS)), dtype=np.float32)

    # 0: is_beaconing - IAT in [1.5, 5] AND Rate in [0.1, 5]
    # c&c-heartbeat: iat median 1.8, rate median 0.55
    C[:, 0] = (
        (df["IAT"].values > 1.5) &
        (df["IAT"].values < 5.0) &
        (df["Rate"].values > 0.1) &
        (df["Rate"].values < 5.0)
    ).astype(np.float32)

    # 1: is_high_rate - Rate > 1000 pps (ddos: median 699,050)
    C[:, 1] = (df["Rate"].values > 1000.0).astype(np.float32)

    # 2: is_syn_heavy - syn_count > 8 (ddos p90=28, c&c-hb p90=12)
    C[:, 2] = (df["syn_count"].values > 8).astype(np.float32)

    # 3: is_short_connection - duration in [0, 0.01) (ddos median=0, benign=-1 missing)
    dur = df["duration"].values
    C[:, 3] = (
        (dur >= 0.0) &
        (dur < 0.01)
    ).astype(np.float32)

    # 4: is_persistent - duration > 2.5 s (c&c-hb and benign long sessions)
    C[:, 4] = (dur > 2.5).astype(np.float32)

    # 5: is_asymmetric - originator sends but no response (resp_pkts == 0 AND orig_pkts > 1)
    C[:, 5] = (
        (df["resp_pkts"].values == 0) &
        (df["orig_pkts"].values > 1)
    ).astype(np.float32)

    # 6: is_single_packet_probe - orig_pkts == 1 (single probe, no sustained data exchange)
    # benign 86.6%, okiru 77.8% -> probe-like; ddos only 1.5%, c&c-hb 45.1%
    C[:, 6] = (df["orig_pkts"].values == 1).astype(np.float32)

    # 7: is_incomplete_handshake - syn without fin (flood / scan pattern)
    # syn_count > 2 AND fin_count == 0: syn packets sent, connection never cleanly closed
    C[:, 7] = (
        (df["syn_count"].values > 2) &
        (df["fin_count"].values == 0)
    ).astype(np.float32)

    return C


# cic-iot-2023 concept rules

CIC_CONCEPTS = [
    "is_high_rate",
    "is_syn_flood",
    "is_udp_dominant",
    "is_short_connection",
    "is_large_payload",
    "is_port_scan",
    "is_high_variance",
    "is_persistent",
]


# returns (N, 8) float32 binary concept labels for cic-iot-2023.
# cic features are z-scored (standardscaler). thresholds are in standardized units.
# derived from per-class distribution analysis:
# ddos-syn_flood:      syn_count z~=2.03, syn_flag z~=1.83, low avg
# mirai-greeth_flood:  avg z~=0.53, not syn-heavy, low tcp
# recon-portscan:      rst_count p90~=2.27, variance p90~=0.26
# vulnerabilityscan:   fin_count p75~=1.37, mixed
# benign:              all near zero
def cic_concept_labels(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    C = np.zeros((n, len(CIC_CONCEPTS)), dtype=np.float32)

    feats = df.columns.tolist()

    def _col(name: str, default=0.0) -> np.ndarray:
        return df[name].values if name in feats else np.full(n, default)

    rate  = _col("rate")
    syn   = _col("syn_count")
    syn_f = _col("syn_flag_number")
    rst   = _col("rst_count")
    fin   = _col("fin_count")
    iat   = _col("iat")             # inter-arrival time (z-scored)
    iat_s = _col("variance")        # variance of IAT -> burstiness
    avg   = _col("avg")             # avg packet size (z-scored)
    udp   = _col("udp")             # UDP packet count (z-scored)
    tcp   = _col("tcp")

    # 0: is_high_rate - rate z > 0.15 (ddos-syn median=0.17, benign median=-0.24)
    C[:, 0] = (rate > 0.15).astype(np.float32)

    # 1: is_syn_flood - syn_count z > 1.5 (ddos-syn: z=2.03 constant; others ~= -0.52)
    C[:, 1] = (syn > 1.5).astype(np.float32)

    # 2: is_udp_dominant - tcp z < -1.0 (mirai: median tcp=-1.67, uses udp instead)
    C[:, 2] = (tcp < -1.0).astype(np.float32)

    # 3: is_short_connection - iat z < -0.003 (ddos-syn: iat median=-0.0038, near minimum)
    # tight cluster at minimum value -> nearly zero inter-arrival time
    C[:, 3] = (iat < -0.0037).astype(np.float32) if "iat" in feats else (
        np.zeros(n, dtype=np.float32)
    )

    # 4: is_large_payload - avg z > 0.4 (mirai: median avg=0.53; ddos-syn: -0.65)
    C[:, 4] = (avg > 0.4).astype(np.float32)

    # 5: is_port_scan - rst_count z > 1.5 (recon p90=2.27; others mostly -0.18)
    C[:, 5] = (rst > 1.5).astype(np.float32)

    # 6: is_high_variance - variance z > 0.5 (benign p90=1.42; ddos-syn locked at -0.21)
    C[:, 6] = (iat_s > 0.5).astype(np.float32)

    # 7: is_persistent - fin_count z > 1.0 (vulnscan p75=1.37; ddos-syn locked at -0.37)
    C[:, 7] = (fin > 1.0).astype(np.float32)

    return C


# quality analysis

# computes:
# - prevalence: fraction of samples where concept=1
# - cross-cutting: each concept appears in >=2 classes (avoids redundancy with label)
# - class-concept correlation matrix (phi coefficient for binary x binary)
# returns dict with all metrics.
def concept_quality_report(
    concept_labels: np.ndarray,
    class_ids: np.ndarray,
    concept_names: list,
    id_to_label: dict,
    dataset: str = "CTU",
) -> dict:
    n_concepts = len(concept_names)
    class_labels = sorted(id_to_label.keys())
    n_classes = len(class_labels)

    prevalence = concept_labels.mean(axis=0)

    # phi correlation matrix: (n_classes, n_concepts)
    phi = np.zeros((n_classes, n_concepts))
    for ci, cid in enumerate(class_labels):
        mask = (class_ids == cid)
        for j in range(n_concepts):
            c = concept_labels[:, j]
            n11 = ( mask &  (c == 1)).sum()
            n10 = ( mask &  (c == 0)).sum()
            n01 = (~mask &  (c == 1)).sum()
            n00 = (~mask &  (c == 0)).sum()
            denom = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
            phi[ci, j] = (n11*n00 - n10*n01) / denom if denom > 0 else 0.0

    # cross-cutting: concept is 1 in >=2 classes (at least 5% prevalence in each)
    cross_cutting = []
    for j in range(n_concepts):
        active_classes = 0
        for ci, cid in enumerate(class_labels):
            mask = (class_ids == cid)
            if concept_labels[mask, j].mean() > 0.05:
                active_classes += 1
        cross_cutting.append(active_classes >= 2)

    report = {
        "dataset": dataset,
        "concept_names": concept_names,
        "prevalence": prevalence.tolist(),
        "cross_cutting": cross_cutting,
        "phi_matrix": phi.tolist(),
        "class_labels": [id_to_label[k] for k in class_labels],
    }
    return report


def print_quality_report(report: dict):
    print(f"\n{'='*60}")
    print(f"  Concept Quality Report - {report['dataset']}")
    print(f"{'='*60}")
    names = report["concept_names"]
    prev  = report["prevalence"]
    cc    = report["cross_cutting"]

    print(f"\n{'Concept':<28}  {'Prev':>6}  {'Cross-cut'}")
    print("-" * 50)
    for i, name in enumerate(names):
        flag = "YES" if cc[i] else "no "
        print(f"  {name:<26}  {prev[i]:>6.3f}  {flag}")

    phi   = np.array(report["phi_matrix"])
    clabs = report["class_labels"]
    print(f"\nPhi correlation (class x concept):")
    header = f"{'':20}" + "".join(f"{n[:8]:>9}" for n in names)
    print(header)
    for ci, clab in enumerate(clabs):
        row = f"  {clab[:18]:<18}" + "".join(f"{phi[ci, j]:>+9.3f}" for j in range(len(names)))
        print(row)


# entry point

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ctu"

    if dataset == "ctu":
        data_dir = Path(__file__).parent.parent / "data"
        vocab    = json.loads((data_dir / "vocab.json").read_text())
        df       = pd.read_parquet(data_dir / "train.parquet")
        labels   = ctu_concept_labels(df)
        id_to_label = {int(k): v for k, v in vocab["id_to_label"].items()
                       if int(k) in vocab["known_ids"]}
        known_mask = df["label_id"].isin(vocab["known_ids"])
        report = concept_quality_report(
            labels[known_mask],
            df["label_id"].values[known_mask],
            CTU_CONCEPTS, id_to_label, "CTU-IoT-23"
        )
        print_quality_report(report)

    elif dataset == "cic":
        data_dir = Path(__file__).parent.parent / "data" / "cic"
        vocab    = json.loads((data_dir / "vocab.json").read_text())
        df       = pd.read_parquet(data_dir / "train.parquet")
        labels   = cic_concept_labels(df)
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]
        id_to_label = {v: k for k, v in l2i.items() if v in known_ids}
        known_mask = df["label_id"].isin(known_ids)
        report = concept_quality_report(
            labels[known_mask],
            df["label_id"].values[known_mask],
            CIC_CONCEPTS, id_to_label, "CIC-IoT-2023"
        )
        print_quality_report(report)
