#!/usr/bin/env python3
"""
Clear all generated *_dpkt.csv files from IoT scenario directories.

Run from ~/Research

Usage:
    python clear_outputs.py
    python clear_outputs.py --dry-run
"""

import os
import glob
import argparse

SCENARIOS_DIR = "data/opt/Malware-Project/BigDataset/IoTScenarios"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be deleted without deleting them"
    )
    args = parser.parse_args()

    if not os.path.exists(SCENARIOS_DIR):
        print(f"ERROR: Directory not found: {SCENARIOS_DIR}")
        return

    # Find all *_dpkt.csv files recursively
    pattern = os.path.join(SCENARIOS_DIR, "**", "*_dpkt.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("No output files found.")
        return

    print(f"Found {len(files)} output files.\n")

    deleted = 0
    for f in files:
        if args.dry_run:
            print(f"[DRY RUN] Would delete: {f}")
        else:
            try:
                os.remove(f)
                print(f"Deleted: {f}")
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

    if args.dry_run:
        print("\nDry run complete. No files were deleted.")
    else:
        print(f"\nDeleted {deleted} files.")


if __name__ == "__main__":
    main()
