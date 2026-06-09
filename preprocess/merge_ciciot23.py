#!/usr/bin/env python3
# merge_ciciot23.py - build cic-iot-2023 duckdb from dpkt csvs.
# uses the duckdb cli binary (not python module) for reliability.
# reads all per-chunk csv files per category, injects label from directory name,
# normalises column names (spaces -> underscores), writes into a single table 'flows'.
#
# output:  ~/Research/work/hyper_iot/data/ciciot23.duckdb
#
# usage:
#   python merge_ciciot23.py [--csv-dir PATH] [--out PATH]

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DUCKDB_BIN  = Path.home() / "Research" / "duckdb"
DEFAULT_CSV = Path.home() / "Research" / "downloads" / "csv"
DEFAULT_OUT = Path.home() / "Research" / "work" / "hyper_iot" / "data" / "ciciot23.duckdb"


# run sql against duckdb cli and return stdout.
def run_sql(db_path: Path, sql: str) -> str:
    result = subprocess.run(
        [str(DUCKDB_BIN), str(db_path)],
        input=sql, text=True, capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"DuckDB error:\n{result.stderr}")
    return result.stdout


def category_dirs(csv_dir: Path) -> list:
    return sorted(p for p in csv_dir.iterdir() if p.is_dir())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv-dir', type=Path, default=DEFAULT_CSV)
    p.add_argument('--out',     type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.csv_dir.exists():
        sys.exit(f"ERROR: CSV dir not found: {args.csv_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.exists():
        print(f"Removing existing database: {args.out}")
        args.out.unlink()

    cats = category_dirs(args.csv_dir)
    print(f"Found {len(cats)} categories")
    print(f"Output : {args.out}\n")

    # build one big sql script: CREATE TABLE from first category,
    # INSERT INTO for all subsequent ones.
    # normalize_names=true: duckdb turns spaces -> underscores in column names.
    sql_parts = []
    for i, cat_dir in enumerate(cats):
        label   = cat_dir.name
        # use glob pattern - duckdb read_csv_auto accepts wildcards
        glob    = str(cat_dir / "*.csv").replace("'", "''")
        select  = f"""
    SELECT *, '{label}' AS label
    FROM read_csv_auto('{glob}', header=true, normalize_names=true,
                       parallel=true, ignore_errors=true)"""

        if i == 0:
            sql_parts.append(f"CREATE TABLE flows AS{select};")
        else:
            sql_parts.append(f"INSERT INTO flows{select};")

    sql_parts.append(
        "SELECT label, COUNT(*) AS n FROM flows GROUP BY label ORDER BY n DESC;"
    )

    full_sql = "\n".join(sql_parts)

    # write sql to temp file so we can inspect on failure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sql',
                                     delete=False, prefix='ciciot23_') as tf:
        tf.write(full_sql)
        sql_file = tf.name

    print(f"Running DuckDB CLI (SQL written to {sql_file}) ...")
    result = subprocess.run(
        [str(DUCKDB_BIN), str(args.out)],
        input=full_sql, text=True, capture_output=True
    )
    if result.returncode != 0:
        print("FAILED. DuckDB stderr:")
        print(result.stderr[-3000:])
        sys.exit(1)

    print(result.stdout)
    print(f"\nDone. Database: {args.out}")

    # quick row count verification
    row_count_sql = "SELECT COUNT(*) AS total FROM flows;"
    out = run_sql(args.out, row_count_sql)
    print(f"Total rows: {out.strip()}")


if __name__ == "__main__":
    main()
