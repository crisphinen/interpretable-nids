# iot-23 zeek + dpkt feature merger, optimized version.
# uses typed csv reads (no all_varchar + try_cast overhead).
# protocol join arm restored: zeek string maps to dpkt integer.
# corrected timestamp window: dpkt ts_start is always before zeek ts_zeek.
# single-scenario mode (--scenario) always overwrites the existing merge.
# qualify clause keeps only the single closest dpkt match per zeek flow uid.
#
# usage:
#   python merge_zeek_dpkt_outer.py                            - skip already-merged
#   python merge_zeek_dpkt_outer.py --reset                    - delete and reprocess all
#   python merge_zeek_dpkt_outer.py --scenario CTU-Honeypot-Capture-4-1

import os
import gc
import glob
import argparse
import duckdb
import pandas as pd

# config
SCENARIOS_DIR       = "data/opt/Malware-Project/BigDataset/IoTScenarios"
MERGED_SUFFIX       = "_merged_outer.csv"
STATS_OUTPUT        = "merge_statistics.csv"
DUCKDB_MEMORY_LIMIT = "100GB"
#DUCKDB_SPILL_DIR    = "/tmp/duckdb_spill"
DUCKDB_SPILL_DIR = "/mnt/678fc01a-1165-4fc8-83f8-481ccd9508b3/Ngari/duckdb_spill"
BROADCAST_PREFIXES = ("224.", "225.", "226.", "227.", "228.", "229.",
                      "230.", "231.", "232.", "233.", "234.", "235.",
                      "236.", "237.", "238.", "239.")
BROADCAST_EXACT    = {"0.0.0.0", "255.255.255.255"}

ZEEK_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p",
    "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp",
    "missed_bytes", "history", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed_label",
]

ZEEK_CLEAN_COLUMNS = [
    "ts_zeek", "uid", "src_ip", "src_port",
    "dst_ip", "dst_port", "protocol_zeek",
    "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp",
    "missed_bytes", "history", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed_label",
]

# typed column definitions for duckdb - avoids all_varchar + try_cast overhead
ZEEK_CSV_COLUMNS = """
    'ts_zeek':       'DOUBLE',
    'uid':           'VARCHAR',
    'src_ip':        'VARCHAR',
    'src_port':      'INTEGER',
    'dst_ip':        'VARCHAR',
    'dst_port':      'INTEGER',
    'protocol_zeek': 'VARCHAR',
    'service':       'VARCHAR',
    'duration':      'DOUBLE',
    'orig_bytes':    'DOUBLE',
    'resp_bytes':    'DOUBLE',
    'conn_state':    'VARCHAR',
    'local_orig':    'VARCHAR',
    'local_resp':    'VARCHAR',
    'missed_bytes':  'DOUBLE',
    'history':       'VARCHAR',
    'orig_pkts':     'DOUBLE',
    'orig_ip_bytes': 'DOUBLE',
    'resp_pkts':     'DOUBLE',
    'resp_ip_bytes': 'DOUBLE',
    'tunnel_parents':'VARCHAR',
    'label':         'VARCHAR',
    'detailed_label':'VARCHAR'
"""

DPKT_CSV_COLUMNS = """
    'src_ip':        'VARCHAR',
    'dst_ip':        'VARCHAR',
    'src_port':      'INTEGER',
    'dst_port':      'INTEGER',
    'protocol':      'INTEGER',
    'ts_start':      'DOUBLE',
    'Number':        'DOUBLE',
    'Tot_size':      'DOUBLE',
    'Min':           'DOUBLE',
    'Max':           'DOUBLE',
    'AVG':           'DOUBLE',
    'Std':           'DOUBLE',
    'Variance':      'DOUBLE',
    'IAT':           'DOUBLE',
    'TTL':           'DOUBLE',
    'Header_Length': 'DOUBLE',
    'fin_count':     'DOUBLE',
    'syn_count':     'DOUBLE',
    'rst_count':     'DOUBLE',
    'ack_count':     'DOUBLE',
    'urg_count':     'DOUBLE',
    'fin_flag_number':'DOUBLE',
    'syn_flag_number':'DOUBLE',
    'rst_flag_number':'DOUBLE',
    'psh_flag_number':'DOUBLE',
    'ack_flag_number':'DOUBLE',
    'ece_flag_number':'DOUBLE',
    'cwr_flag_number':'DOUBLE',
    'TCP':           'DOUBLE',
    'UDP':           'DOUBLE',
    'ICMP':          'DOUBLE',
    'IGMP':          'DOUBLE',
    'ARP':           'DOUBLE',
    'IPv':           'DOUBLE',
    'LLC':           'DOUBLE',
    'HTTP':          'DOUBLE',
    'HTTPS':         'DOUBLE',
    'DNS':           'DOUBLE',
    'SSH':           'DOUBLE',
    'Telnet':        'DOUBLE',
    'SMTP':          'DOUBLE',
    'IRC':           'DOUBLE',
    'DHCP':          'DOUBLE',
    'Rate':          'DOUBLE',
    'Srate':         'DOUBLE',
    'Drate':         'DOUBLE',
    'Magnitude':     'DOUBLE',
    'Radius':        'DOUBLE',
    'Weight':        'DOUBLE',
    'Covariance':    'DOUBLE'
"""


# logging
def log(msg):
    print(msg, flush=True)


# helpers
def find_conn_logs(capture_dir):
    found  = glob.glob(os.path.join(capture_dir, "bro", "conn.log.labeled"))
    found += glob.glob(os.path.join(capture_dir, "*", "bro", "conn.log.labeled"))
    return sorted(found)


def find_dpkt_csv(capture_dir):
    name = os.path.basename(capture_dir)
    path = os.path.join(capture_dir, f"{name}_dpkt.csv")
    return path if os.path.exists(path) else None


def is_broadcast(ip):
    if not ip or ip == "-":
        return True
    if ip in BROADCAST_EXACT:
        return True
    for prefix in BROADCAST_PREFIXES:
        if ip.startswith(prefix):
            return True
    return False


def parse_zeek_line(line):
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) < len(ZEEK_COLUMNS):
        tail = parts[-1].split()
        if len(tail) >= 3:
            parts = parts[:-1] + [tail[0], tail[1], tail[2]]
        elif len(tail) == 2:
            parts = parts[:-1] + [tail[0], tail[1], ""]
        elif len(tail) == 1:
            parts = parts[:-1] + [tail[0], "", ""]
    while len(parts) < len(ZEEK_COLUMNS):
        parts.append("")
    return parts[:len(ZEEK_COLUMNS)]


def preprocess_zeek_to_csv(conn_log_paths, out_path):
    import csv

    total_raw    = 0
    total_clean  = 0
    label_counts = {}

    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(ZEEK_CLEAN_COLUMNS)

        for log_path in conn_log_paths:
            log(f"  Preprocessing: {os.path.basename(log_path)}")
            with open(log_path, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    total_raw += 1

                    fields = parse_zeek_line(line)
                    src_ip = fields[2].strip()
                    dst_ip = fields[4].strip()

                    if is_broadcast(src_ip) or is_broadcast(dst_ip):
                        continue

                    fields = ["" if f == "-" else f for f in fields]
                    total_clean += 1
                    label = fields[21] if len(fields) > 21 else ""
                    label_counts[label] = label_counts.get(label, 0) + 1
                    writer.writerow(fields)

                    if total_raw % 5_000_000 == 0:
                        log(f"    ... {total_raw:,} lines read, "
                            f"{total_clean:,} kept so far")

    return total_raw, total_clean, label_counts


# join sql
JOIN_SQL = """
COPY (
    SELECT
        -- Key fields: prefer Zeek side
        COALESCE(z.src_ip,   d.src_ip)   AS src_ip,
        COALESCE(z.dst_ip,   d.dst_ip)   AS dst_ip,
        COALESCE(z.src_port, d.src_port) AS src_port,
        COALESCE(z.dst_port, d.dst_port) AS dst_port,

        -- Zeek flow fields
        z.protocol_zeek                  AS protocol,
        z.ts_zeek                        AS ts,
        z.duration,
        z.conn_state,
        z.history,
        z.local_orig,
        z.local_resp,
        z.missed_bytes,
        z.orig_bytes,
        z.orig_ip_bytes,
        z.orig_pkts,
        z.service,
        z.resp_bytes,
        z.resp_ip_bytes,
        z.resp_pkts,
        z.tunnel_parents,
        z.label,
        z.detailed_label,

        -- DPKT packet-level features (NULL for Zeek-only rows)
        d.Header_Length,
        d.protocol                       AS Protocol_Type,
        d.TTL                            AS Time_To_Live,
        d.Rate,
        d.fin_flag_number,
        d.syn_flag_number,
        d.rst_flag_number,
        d.psh_flag_number,
        d.ack_flag_number,
        d.ece_flag_number,
        d.cwr_flag_number,
        d.ack_count,
        d.syn_count,
        d.fin_count,
        d.rst_count,
        d.HTTP, d.HTTPS, d.DNS, d.Telnet, d.SMTP,
        d.SSH,  d.IRC,   d.TCP, d.UDP,    d.DHCP,
        d.ARP,  d.ICMP,  d.IGMP, d.IPv,  d.LLC,
        d.Tot_size                       AS Tot_sum,
        d.Min, d.Max, d.AVG, d.Std, d.Tot_size,
        d.IAT, d.Number, d.Variance,
        d.Srate, d.Drate, d.Magnitude, d.Radius,
        d.Weight, d.Covariance,
        d.ts_start                       AS dpkt_ts_start,

        -- Derived features from Zeek
        z.orig_bytes / (z.resp_bytes + 1)                AS byte_ratio,
        z.orig_pkts  / (z.duration   + 1)                AS orig_pkt_rate,
        z.orig_ip_bytes / (z.duration + 1)               AS orig_byte_rate,
        z.orig_bytes - z.resp_bytes                      AS direction

    FROM read_csv('{zeek_csv}',
            header=true,
            ignore_errors=true,
            columns={{{zeek_cols}}}) z

    FULL OUTER JOIN read_csv('{dpkt_csv}',
            header=true,
            ignore_errors=true,
            columns={{{dpkt_cols}}}) d

        ON  z.src_ip   = d.src_ip
        AND z.dst_ip   = d.dst_ip
        AND z.src_port = d.src_port
        AND z.dst_port = d.dst_port

        -- Protocol: map Zeek string to DPKT integer
        AND CASE LOWER(z.protocol_zeek)
                WHEN 'tcp'  THEN 6
                WHEN 'udp'  THEN 17
                WHEN 'icmp' THEN 1
                ELSE 0
            END = d.protocol

        -- Timestamp: DPKT ts_start is always <= Zeek ts_zeek (first packet vs flow end)
        -- Window: look back by flow duration + 60s buffer, forward 60s for clock skew
        AND d.ts_start BETWEEN z.ts_zeek - COALESCE(z.duration, 0) - 60
                           AND z.ts_zeek + 60

    -- Drop DPKT-only rows: no Zeek label means unusable for training
    WHERE z.label IS NOT NULL

    -- For each Zeek flow, keep only the single closest DPKT match by timestamp.
    -- Prevents duplicates when DPKT splits long flows at the 60s timeout boundary.
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY z.uid
        ORDER BY ABS(d.ts_start - z.ts_zeek)
    ) = 1

) TO '{out_csv}' (HEADER, DELIMITER ',');
"""


# per-scenario merge
def merge_scenario(capture_dir, con, reset=False, force_overwrite=False):
    name     = os.path.basename(capture_dir)
    out_path = os.path.join(capture_dir, f"{name}{MERGED_SUFFIX}")

    if os.path.exists(out_path):
        if force_overwrite or reset:
            os.remove(out_path)
            log(f"[{name}] Deleted existing merged CSV")
        else:
            log(f"[{name}] SKIPPED - already merged")
            return None

    conn_logs = find_conn_logs(capture_dir)
    dpkt_path = find_dpkt_csv(capture_dir)

    if not conn_logs:
        log(f"[{name}] SKIPPED - no conn.log.labeled found")
        return None
    if not dpkt_path:
        log(f"[{name}] SKIPPED - no _dpkt.csv found")
        return None

    zeek_size = sum(os.path.getsize(p) for p in conn_logs) / (1024**3)
    dpkt_size = os.path.getsize(dpkt_path) / (1024**3)
    log(f"[{name}] Zeek: {zeek_size:.2f} GB | DPKT: {dpkt_size:.2f} GB")

    tmp_zeek = os.path.join(capture_dir, f"{name}_zeek_clean.tmp.csv")
    tmp_out  = out_path + ".tmp"

    for p in (tmp_zeek, tmp_out):
        if os.path.exists(p):
            os.remove(p)

    try:
        log(f"[{name}] Step 1/2: Preprocessing Zeek log...")
        total_raw, total_clean, label_counts = preprocess_zeek_to_csv(
            conn_logs, tmp_zeek
        )
        tmp_zeek_size = os.path.getsize(tmp_zeek) / (1024**3)
        log(f"[{name}] {total_raw:,} raw -> {total_clean:,} clean "
            f"(temp CSV: {tmp_zeek_size:.2f} GB)")

        if total_clean == 0:
            log(f"[{name}] SKIPPED - no rows after broadcast filter")
            return None

        log(f"[{name}] Step 2/2: DuckDB join...")
        sql = JOIN_SQL.format(
            zeek_csv  = os.path.abspath(tmp_zeek),
            dpkt_csv  = os.path.abspath(dpkt_path),
            out_csv   = os.path.abspath(tmp_out),
            zeek_cols = ZEEK_CSV_COLUMNS,
            dpkt_cols = DPKT_CSV_COLUMNS,
        )
        con.execute(sql)
        log(f"[{name}] DuckDB join complete")

    except Exception as e:
        log(f"[{name}] ERROR: {e}")
        for p in (tmp_zeek, tmp_out):
            if os.path.exists(p):
                os.remove(p)
        return None
    finally:
        if os.path.exists(tmp_zeek):
            os.remove(tmp_zeek)

    if not os.path.exists(tmp_out):
        log(f"[{name}] WARNING - output not created (0 matches)")
        merged_rows = 0
    else:
        merged_rows = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{os.path.abspath(tmp_out)}', "
            f"header=true, ignore_errors=true)"
        ).fetchone()[0]

    if merged_rows > 0:
        os.rename(tmp_out, out_path)
    else:
        log(f"[{name}] WARNING - merge produced 0 rows")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

    # label stats from merged output
    merged_benign = merged_malicious = merged_unique = 0
    if merged_rows > 0:
        lbl_df = con.execute(
            f"SELECT label, COUNT(*) AS cnt "
            f"FROM read_csv_auto('{os.path.abspath(out_path)}', header=true, ignore_errors=true) "
            f"GROUP BY label"
        ).df()
        merged_benign    = int(lbl_df.loc[lbl_df["label"] == "benign", "cnt"].sum())
        merged_malicious = int(lbl_df.loc[lbl_df["label"] != "benign", "cnt"].sum())
        merged_unique    = len(lbl_df)

    dpkt_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{os.path.abspath(dpkt_path)}', "
        f"header=true, ignore_errors=true)"
    ).fetchone()[0]

    stats = {
        "scenario":                     name,
        "zeek_flows_total":             total_raw,
        "zeek_flows_after_filter":      total_clean,
        "zeek_flows_dropped_broadcast": total_raw - total_clean,
        "zeek_benign":                  label_counts.get("benign", 0),
        "zeek_malicious":               sum(
            v for k, v in label_counts.items() if k != "benign"
        ),
        "zeek_unique_labels":           len(label_counts),
        "dpkt_rows":                    dpkt_rows,
        "merged_rows":                  merged_rows,
        "merge_match_rate_pct":         round(
            100 * merged_rows / total_clean if total_clean > 0 else 0, 2
        ),
        "merged_benign":                merged_benign,
        "merged_malicious":             merged_malicious,
        "merged_unique_labels":         merged_unique,
    }

    print_scenario_stats(stats)
    gc.collect()
    return stats


# display
def print_scenario_stats(stats):
    log(f"\n  {'-'*57}")
    log(f"  Scenario : {stats['scenario']}")
    log(f"  {'-'*57}")
    log(f"  Zeek flows (raw)          : {stats['zeek_flows_total']:>12,}")
    log(f"  Zeek after broadcast drop : {stats['zeek_flows_after_filter']:>12,}  "
        f"(dropped {stats['zeek_flows_dropped_broadcast']:,})")
    log(f"    Benign                  : {stats['zeek_benign']:>12,}")
    log(f"    Malicious               : {stats['zeek_malicious']:>12,}")
    log(f"    Unique labels           : {stats['zeek_unique_labels']:>12,}")
    log(f"  DPKT rows                 : {stats['dpkt_rows']:>12,}")
    log(f"  Merged rows               : {stats['merged_rows']:>12,}  "
        f"({stats['merge_match_rate_pct']}% match rate)")
    log(f"    Benign                  : {stats['merged_benign']:>12,}")
    log(f"    Malicious               : {stats['merged_malicious']:>12,}")
    log(f"    Unique labels           : {stats['merged_unique_labels']:>12,}")


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing merged CSVs and reprocess all.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Single scenario name - always overwrites existing merge.")
    args = parser.parse_args()

    # single-scenario mode
    if args.scenario:
        capture_path = os.path.join(SCENARIOS_DIR, args.scenario)
        if not os.path.isdir(capture_path):
            log(f"ERROR: scenario directory not found: {capture_path}")
            return
        all_captures    = [capture_path]
        force_overwrite = True

    # bulk mode
    else:
        all_captures = sorted([
            d for d in glob.glob(os.path.join(SCENARIOS_DIR, "CTU-*"))
            if os.path.isdir(d)
        ])
        force_overwrite = False

    if not all_captures:
        log(f"ERROR: No capture directories found in {SCENARIOS_DIR}")
        return

    if args.reset and not args.scenario:
        log("WARNING: --reset flag set - all existing merged CSVs will be deleted.")

    log(f"Found {len(all_captures)} capture director(ies)")
    log(f"Output: one *{MERGED_SUFFIX} per scenario\n")

    os.makedirs(DUCKDB_SPILL_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{DUCKDB_SPILL_DIR}'")

    all_stats = []

    for i, capture_dir in enumerate(all_captures, 1):
        name = os.path.basename(capture_dir)
        log(f"\n{'='*59}")
        log(f"  [{i}/{len(all_captures)}] {name}")
        log(f"{'='*59}")

        stats = merge_scenario(
            capture_dir, con,
            reset=args.reset,
            force_overwrite=force_overwrite
        )
        if stats is not None:
            all_stats.append(stats)
        gc.collect()

    con.close()

    if not all_stats:
        log("\nNo scenarios were merged.")
        return

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_OUTPUT, index=False)

    log(f"\n{'='*59}")
    log(f"  GLOBAL SUMMARY - {len(all_stats)} scenarios processed")
    log(f"{'='*59}")
    log(f"  Total Zeek flows (raw)      : {stats_df['zeek_flows_total'].sum():>14,}")
    log(f"  Total after broadcast drop  : {stats_df['zeek_flows_after_filter'].sum():>14,}")
    log(f"  Total DPKT rows             : {stats_df['dpkt_rows'].sum():>14,}")
    log(f"  Total merged rows           : {stats_df['merged_rows'].sum():>14,}")
    log(f"  Total merged benign         : {stats_df['merged_benign'].sum():>14,}")
    log(f"  Total merged malicious      : {stats_df['merged_malicious'].sum():>14,}")
    log(f"  Avg match rate              : {stats_df['merge_match_rate_pct'].mean():>13.1f}%")
    log(f"\n  Full statistics -> {STATS_OUTPUT}")
    log(f"{'='*59}\n")


if __name__ == "__main__":
    main()
