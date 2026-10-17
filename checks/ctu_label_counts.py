#!/usr/bin/env python3
# CTU-IoT-23 data documentation (supplement, Feature Derivation): what the evaluated flow table holds.
#   - the label filter and zero-fill rules stored in the DuckDB views (cleaned, preprocessed);
#   - retained flows per label in preprocessed_sorted, and how many of them had no matching
#     packet-derived (dpkt) flow: the merge keeps such Zeek records and preprocessing fills their
#     packet features with 0 (Protocol_Type, Header_Length and Tot_sum are all 0);
#   - groups of rows sharing every Zeek field (one Zeek record giving several rows);
#   - the zero-filled share in each evaluated split (data/ctuk).
# Writes results/cbm/ctu_label_counts.json.
#
# usage: python checks/ctu_label_counts.py
import json, sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DB_PATH

ZERO = "Protocol_Type = 0 AND Header_Length = 0 AND Tot_sum = 0"


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    q = lambda s: con.execute(s).fetchall()
    views = {n: s for n, s in q("SELECT view_name, sql FROM duckdb_views() WHERE NOT internal")}
    rows = q(f"""SELECT detailed_label, COUNT(*), SUM(CASE WHEN {ZERO} THEN 1 ELSE 0 END)
                 FROM preprocessed_sorted GROUP BY 1 ORDER BY 2 DESC""")
    per_label = {l: {"retained": int(n), "no_packet_match": int(z)} for l, n, z in rows}
    dup = q("""SELECT COUNT(*), COALESCE(SUM(c - 1), 0) FROM (
                 SELECT COUNT(*) c FROM preprocessed_sorted
                 GROUP BY src_ip, ts, duration, orig_bytes, resp_bytes, history, detailed_label HAVING COUNT(*) > 1)""")[0]
    proto = {str(p): int(n) for p, n in q("SELECT Protocol_Type, COUNT(*) FROM preprocessed_sorted GROUP BY 1")}
    splits = {}
    for s in ("train", "val", "test_known", "test_unknown"):
        d = pd.read_parquet(ROOT / "data/ctuk" / f"{s}.parquet")
        z = (d.Protocol_Type == 0) & (d.Header_Length == 0) & (d.Tot_sum == 0)
        splits[s] = {"rows": int(len(d)), "no_packet_match": int(z.sum()),
                     "by_label": {k: int(v) for k, v in d[z].detailed_label.value_counts().items()}}
    out = {"views": views, "per_label": per_label, "protocol_type_counts": proto,
           "duplicate_zeek_groups": {"groups": int(dup[0]), "extra_rows": int(dup[1])}, "ctuk_splits": splits}
    (ROOT / "results/cbm/ctu_label_counts.json").write_text(json.dumps(out, indent=2))
    tot = sum(v["retained"] for v in per_label.values()); tz = sum(v["no_packet_match"] for v in per_label.values())
    print(f"retained {tot:,} flows, {tz:,} without a packet match ({tz / tot:.1%})")
    for l, v in per_label.items():
        print(f"  {l:28s} {v['retained']:>12,}  no packet match {v['no_packet_match']:>10,} ({v['no_packet_match'] / v['retained']:.1%})")
    print("duplicate Zeek groups:", out["duplicate_zeek_groups"])
    for s, v in splits.items():
        print(f"  ctuk {s:12s} {v['no_packet_match']:>5} of {v['rows']:,} without a packet match {v['by_label']}")


if __name__ == "__main__":
    main()
