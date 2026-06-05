import numpy as np
import pandas as pd
from pathlib import Path
import json

# CTU-IoT-23 concept rules

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


def ctu_concept_labels(df: pd.DataFrame) -> np.ndarray:
    """Returns (N, 8) float32 binary concept labels for CTU-IoT-23."""
    n = len(df)
    C = np.zeros((n, len(CTU_CONCEPTS)), dtype=np.float32)

    # 0: is_beaconing — periodic C&C heartbeat: IAT in [1.0, 6.0] AND moderate rate AND
    #    multi-packet exchange (orig_pkts > 1 excludes Benign single-packet flows at IAT~1.8s).
    #    Previous rule [1.5,5] fired for 26.9% Benign (IAT~1.8 overlap); orig_pkts>1 cuts this.
    C[:, 0] = (
        (df["IAT"].values > 1.0) &
        (df["IAT"].values < 6.0) &
        (df["Rate"].values > 0.05) &
        (df["Rate"].values < 20.0) &
        (df["orig_pkts"].values > 1)
    ).astype(np.float32)

    # 1: is_high_rate — Rate > 1000 pps (DDoS: median 699,050)
    C[:, 1] = (df["Rate"].values > 1000.0).astype(np.float32)

    # 2: is_syn_heavy — syn_count > 4 (was > 8; catches C&C-HB p75=6, DDoS p50+).
    #    Old threshold missed ~80% of C&C-HB syn-heavy flows.
    C[:, 2] = (df["syn_count"].values > 4).astype(np.float32)

    # 3: is_short_connection — duration in [0, 0.01) (DDoS median=0, Okiru median=0)
    dur = df["duration"].values
    C[:, 3] = (
        (dur >= 0.0) &
        (dur < 0.01)
    ).astype(np.float32)

    # 4: is_persistent — long-lived multi-packet session: duration > 2.5 AND orig_pkts > 1.
    #    orig_pkts > 1 excludes Benign single-packet flows that happen to have long durations.
    C[:, 4] = (
        (dur > 2.5) &
        (df["orig_pkts"].values > 1)
    ).astype(np.float32)

    # 5: is_asymmetric — originator sends but no response (resp_pkts == 0 AND orig_pkts > 1)
    C[:, 5] = (
        (df["resp_pkts"].values == 0) &
        (df["orig_pkts"].values > 1)
    ).astype(np.float32)

    # 6: is_single_packet_probe — orig_pkts == 1 (single-direction probe, no sustained exchange).
    # NOTE: fires for Benign (86.6%) AND Okiru (77.8%) — both have single-packet sentinel flows
    # (dur=-1, Rate=0, resp_pkts=0) that are structurally identical in this feature set.
    # This is a "no-bidirectional-data" concept, not an Okiru-exclusive concept.
    # The other concepts (is_high_rate, is_incomplete_handshake) provide class discrimination.
    C[:, 6] = (df["orig_pkts"].values == 1).astype(np.float32)

    # 7: is_incomplete_handshake — SYN without FIN (flood / scan pattern)
    C[:, 7] = (
        (df["syn_count"].values > 2) &
        (df["fin_count"].values == 0)
    ).astype(np.float32)

    return C


# CIC-IoT-2023 concept rules

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


def cic_concept_labels(df: pd.DataFrame) -> np.ndarray:
    """Returns (N, 8) float32 binary concept labels for CIC-IoT-2023 (z-scored features)."""
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
    iat_s = _col("variance")        # variance of IAT → burstiness
    avg   = _col("avg")             # avg packet size (z-scored)
    udp   = _col("udp")             # UDP packet count (z-scored)
    tcp   = _col("tcp")

    # 0: is_high_rate — rate z > 0.15 (DDoS-SYN median=0.17, Benign median=-0.24)
    C[:, 0] = (rate > 0.15).astype(np.float32)

    # 1: is_syn_flood — syn_count z > 1.5 (DDoS-SYN: z=2.03 constant; others ≈ -0.52)
    C[:, 1] = (syn > 1.5).astype(np.float32)

    # 2: is_udp_dominant — tcp z < -1.0 (Mirai: median tcp=-1.67, uses UDP instead)
    C[:, 2] = (tcp < -1.0).astype(np.float32)

    # 3: is_short_connection — iat z < -0.00375 (DDoS-SYN/Mirai locked at -0.0038;
    #    Benign p5=-0.0038 but p25=-0.0037, so -0.00375 sits between Benign p5 and floor).
    #    Previous threshold -0.0037 was too close to minimum, catching 5% of Benign.
    C[:, 3] = (iat < -0.00375).astype(np.float32) if "iat" in feats else (
        np.zeros(n, dtype=np.float32)
    )

    # 4: is_large_payload — Mirai UDP flood: avg z > 0.4 AND tcp z < -0.5.
    #    tcp < -0.5 excludes Benign large-payload flows (which use TCP, tcp~0);
    #    Mirai: median tcp=-1.67. Benign p75 avg=1.59 caused 26% false positives.
    C[:, 4] = ((avg > 0.4) & (tcp < -0.5)).astype(np.float32)

    # 5: is_port_scan — rst_count z > 1.5 (Recon p90=2.27; others mostly -0.18)
    C[:, 5] = (rst > 1.5).astype(np.float32)

    # 6: is_high_variance — bursty/interactive traffic: variance z > 0.2.
    #    Previous threshold 0.5 was at Benign p75 — only caught top 25% of bursty Benign.
    #    Lowered to 0.2 to capture Benign bursty behavior more reliably (p50=-0.062, p75=0.535).
    #    DDoS-SYN locked at -0.2117; Mirai at -0.2117 to -0.20. Attack floor is ~-0.21.
    C[:, 6] = (iat_s > 0.2).astype(np.float32)

    # 7: is_persistent — fin_count z > 1.0 (VulnScan p75=1.37; DDoS-SYN locked at -0.37)
    C[:, 7] = (fin > 1.0).astype(np.float32)

    return C


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

    # Phi correlation matrix: (n_classes, n_concepts)
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

    # Cross-cutting: concept is 1 in >=2 classes (at least 5% prevalence in each)
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
    print(f"  Concept Quality Report — {report['dataset']}")
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
    print(f"\nPhi correlation (class × concept):")
    header = f"{'':20}" + "".join(f"{n[:8]:>9}" for n in names)
    print(header)
    for ci, clab in enumerate(clabs):
        row = f"  {clab[:18]:<18}" + "".join(f"{phi[ci, j]:>+9.3f}" for j in range(len(names)))
        print(row)


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
