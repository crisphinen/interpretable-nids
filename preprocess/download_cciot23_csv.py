# cic-iot-2023 csv download script.
# downloads all csv files for each category.
#
# completeness logic:
#   if a chunk file already exists on disk: skip it.
#   if a chunk file is missing: download it.
#   use --force (with --category) to re-download all chunks regardless.
#
# usage:
#   python csv_downloader.py                                    - download all eligible categories
#   python csv_downloader.py --retry-failed                     - retry only categories in errors.json
#   python csv_downloader.py --dry-run                          - show what would be downloaded
#   python csv_downloader.py --max-chunks 10                    - override chunk limit (default 5)
#   python csv_downloader.py --category Backdoor_Malware        - single category
#   python csv_downloader.py --category Backdoor_Malware --force  - force re-download

import json
import argparse
import threading
import time
import urllib3
import urllib.parse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# config - edit before running
COOKIE_TOKEN    = "nqterct6q3o3ho944qq0ugpdpq"   # refresh if session expires
CSV_DIR         = Path("downloads/csv")           # where CSVs are saved
PROGRESS_FILE   = Path("progress_csv.txt")        # completed categories
ERROR_FILE      = Path("errors_csv.json")         # failed categories (JSON)

MAX_CHUNKS      = 50        # skip categories with more chunks than this
MAX_RETRIES     = 5        # attempts per category before giving up
RETRY_WAIT      = 10       # seconds between retries
READ_BUFFER     = 1024 * 1024   # 1 MB read buffer

BASE_URL = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file="

HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "accept-language": "en-US,en;q=0.9",
    "connection": "keep-alive",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
    ),
}

# All dataset entries: (category, base_filename, n_chunks, folder_override)
# n_chunks = number of EXTRA chunks after the base file
# e.g. n_chunks=3 means: base.pcap.csv, base1.pcap.csv, base2.pcap.csv, base3.pcap.csv (4 files total)
# folder_override = exact server folder name if it differs from category (None = use category as-is)
ENTRIES = [
    ("Backdoor_Malware",         "Backdoor_Malware",          0,  None),
    ("Benign_Final",             "BenignTraffic",             3,  None),
    ("BrowserHijacking",         "BrowserHijacking",          0,  None),
    ("CommandInjection",         "CommandInjection",          0,  None),
    ("DDoS-ACK_Fragmentation",   "DDoS-ACK_Fragmentation",   12,  None),
    ("DDoS-HTTP_Flood",          "DDoS-HTTP_Flood-",          0,  None),
    ("DDoS-ICMP_Flood",          "DDoS-ICMP_Flood",          26,  None),
    ("DDoS-ICMP_Fragmentation",  "DDoS-ICMP_Fragmentation",  19,  None),
    ("DDoS-PSHACK_Flood",        "DDoS-PSHACK_Flood",        15,  "DDoS-PSHACK_FLOOD"),
    ("DDoS-RSTFINFlood",         "DDoS-RSTFINFlood",         15,  "DDoS-RSTFINFLOOD"),
    ("DDoS-SYN_Flood",           "DDoS-SYN_Flood",           15,  None),
    ("DDoS-SlowLoris",           "DDoS-SlowLoris",            0,  None),
    ("DDoS-SynonymousIP_Flood",  "DDoS-SynonymousIP_Flood",  13,  None),
    ("DDoS-TCP_Flood",           "DDoS-TCP_Flood",           17,  None),
    ("DDoS-UDP_Flood",           "DDoS-UDP_Flood",           20,  None),
    ("DDoS-UDP_Fragmentation",   "DDoS-UDP_Fragmentation",   12,  None),
    ("DNS_Spoofing",             "DNS_Spoofing",               0,  None),
    ("DictionaryBruteForce",     "DictionaryBruteForce",      0,  None),
    ("DoS-HTTP_Flood",           "DoS-HTTP_Flood",             1,  None),
    ("DoS-SYN_Flood",            "DoS-SYN_Flood",              7,  None),
    ("DoS-TCP_Flood",            "DoS-TCP_Flood",             10,  None),
    ("DoS-UDP_Flood",            "DoS-UDP_Flood",             16,  None),
    ("MITM-ArpSpoofing",         "MITM-ArpSpoofing",          1,  None),
    ("Mirai-greeth_flood",       "Mirai-greeth_flood",        28,  None),
    ("Mirai-greip_flood",        "Mirai-greip_flood",         21,  None),
    ("Mirai-udpplain",           "Mirai-udpplain",            24,  None),
    ("Recon-HostDiscovery",      "Recon-HostDiscovery",        0,  None),
    ("Recon-OSScan",             "Recon-OSScan",               0,  None),
    ("Recon-PingSweep",          "Recon-PingSweep",            0,  None),
    ("Recon-PortScan",           "Recon-PortScan",             0,  None),
    ("SqlInjection",             "SqlInjection",               0,  None),
    ("Uploading_Attack",         "Uploading_Attack",           0,  None),
    ("VulnerabilityScan",        "VulnerabilityScan",          0,  None),
    ("XSS",                      "XSS",                        0,  None),
]

# urllib3 pool
_http = urllib3.PoolManager(
    num_pools=4,
    maxsize=2,
    retries=False,
    timeout=urllib3.Timeout(connect=30, read=120),
)

# thread-safe print
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# helpers

# returns list of chunk filenames: [base.pcap.csv, base1.pcap.csv, ..., baseN.pcap.csv]
def build_chunks(fname: str, n: int) -> List[str]:
    files = [f"{fname}.pcap.csv"]
    files += [f"{fname}{i}.pcap.csv" for i in range(1, n + 1)]
    return files


def build_url(category: str, folder: str, filename: str) -> str:
    encoded = urllib.parse.quote(f"CSV/CSV/{folder}/{filename}", safe="")
    return BASE_URL + encoded


def build_referer(folder: str) -> str:
    encoded = urllib.parse.quote(f"CSV/CSV/{folder}/", safe="")
    return f"https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file={encoded}"


# progress and error tracking

def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    return set(l.strip() for l in PROGRESS_FILE.read_text().splitlines() if l.strip())


def mark_done(category: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(category + "\n")
    errors = read_errors()
    if category in errors:
        del errors[category]
        write_errors(errors)


def read_errors() -> dict:
    if not ERROR_FILE.exists():
        return {}
    try:
        return json.loads(ERROR_FILE.read_text())
    except Exception:
        return {}


def write_errors(errors: dict):
    ERROR_FILE.write_text(json.dumps(errors, indent=2))


def log_error(category: str, message: str):
    errors = read_errors()
    existing = errors.get(category, {})
    errors[category] = {
        "last_error": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempts": existing.get("attempts", 0) + 1,
    }
    write_errors(errors)


def load_errors() -> set:
    return set(read_errors().keys())


# completeness check

# determine which chunks need downloading.
# rules:
#   force=True : always download all chunks.
#   force=False: download only chunks whose file does not exist on disk.
def chunks_needing_download(category: str, filenames: List[str], force: bool = False) -> List[Tuple[str, bool]]:
    result = []
    for fname in filenames:
        p = CSV_DIR / category / fname
        if force or not p.exists():
            result.append((fname, True))
        else:
            result.append((fname, False))
    return result


# download

# download a single file from url to dest.
# always overwrites dest if it exists.
# updates pbar with bytes downloaded.
# returns (success, error_message).
def download_file(url: str, dest: Path, referer: str,
                  pbar: tqdm) -> Tuple[bool, str]:
    headers = dict(HEADERS)
    headers["cookie"] = f"Token={COOKIE_TOKEN}"
    headers["referer"] = referer

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = _http.request(
            "GET", url,
            headers=headers,
            preload_content=False,
        )

        if resp.status == 404:
            resp.drain_conn()
            return False, f"HTTP 404 - file does not exist on server"

        if resp.status != 200:
            resp.drain_conn()
            return False, f"HTTP {resp.status}"

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(READ_BUFFER)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

        resp.drain_conn()
        return True, ""

    except Exception as e:
        return False, str(e)


# download all chunks that need downloading for a category.
# returns (success, error_message).
def download_category(category: str, folder: str, filenames: List[str], force: bool = False) -> Tuple[bool, str]:
    referer = build_referer(folder)
    to_download = chunks_needing_download(category, filenames, force=force)

    needs   = [(f, nd) for f, nd in to_download if nd]
    skipped = [(f, nd) for f, nd in to_download if not nd]

    if skipped:
        tprint(f"  ok: skipping {len(skipped)} chunk(s) already on disk")

    if not needs:
        tprint(f"  ok: all chunks already on disk - nothing to download")
        return True, ""

    tprint(f"  downloading {len(needs)} chunk(s) for [{category}] ...")

    with tqdm(
        total=None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=category,
        leave=False,
    ) as pbar:
        for fname, _ in needs:
            url  = build_url(category, folder, fname)
            dest = CSV_DIR / category / fname

            if dest.exists():
                dest.unlink()

            ok, err = download_file(url, dest, referer, pbar)
            if not ok:
                return False, f"Failed to download [{fname}]: {err}"

    tprint(f"  ok: all chunks downloaded for [{category}]")
    return True, ""


# main

def parse_args():
    p = argparse.ArgumentParser(description="CIC-IoT-2023 CSV Downloader")
    p.add_argument("--retry-failed", action="store_true",
                   help="Only retry categories currently in errors_csv.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded without downloading anything")
    p.add_argument("--category", metavar="NAME",
                   help="Download a single specific category")
    p.add_argument("--force", action="store_true",
                   help="Re-download all chunks even if they already exist on disk (use with --category)")
    p.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                   help=f"Skip categories with more chunks than this (default: {MAX_CHUNKS})")
    return p.parse_args()


def main():
    args = parse_args()
    max_chunks = args.max_chunks
    force = args.force

    if force and not args.category:
        print("  warning: --force only applies when used with --category. Ignoring.")
        force = False

    tprint("=" * 60)
    tprint("  CIC-IoT-2023  CSV Downloader")
    tprint("=" * 60)
    tprint(f"  CSV dir        : {CSV_DIR}")
    tprint(f"  Progress file  : {PROGRESS_FILE}")
    tprint(f"  Max chunks     : {max_chunks}")
    if force:
        tprint(f"  Force mode     : ON (will re-download all chunks)")
    tprint()

    # build full category list with chunk filenames
    all_categories = [
        (cat, build_chunks(fname, n), n + 1, folder or cat)
        for cat, fname, n, folder in ENTRIES
    ]
    cat_lookup = {cat: (cat, chunks, total, folder) for cat, chunks, total, folder in all_categories}

    completed = load_progress()
    errored   = load_errors()

    # build pending list
    if args.category:
        if args.category not in cat_lookup:
            tprint(f"  error: unknown category: '{args.category}'")
            tprint(f"  valid categories:")
            for cat, _, _, _ in all_categories:
                tprint(f"    {cat}")
            return
        pending = [cat_lookup[args.category]]

    elif args.retry_failed:
        if not errored:
            tprint("  ok: no failed categories in errors_csv.json")
            return
        pending = [cat_lookup[cat] for cat in errored if cat in cat_lookup]
        tprint(f"  retrying {len(pending)} failed category/categories")

    else:
        pending = [
            (cat, chunks, total, folder)
            for cat, chunks, total, folder in all_categories
            if cat not in completed
        ]

    # apply chunk limit filter
    skipped_by_limit = [(cat, total) for cat, chunks, total, folder in pending if total > max_chunks]
    pending = [(cat, chunks, total, folder) for cat, chunks, total, folder in pending if total <= max_chunks]

    # dry run
    if args.dry_run:
        tprint(f"  DRY RUN - nothing will be downloaded\n")
        tprint(f"  {'CATEGORY':<35} {'CHUNKS':>6}  ACTION")
        tprint(f"  {'-'*35} {'-'*6}  ------")

        for cat, chunks, total, folder in all_categories:
            if cat in completed:
                tprint(f"  {cat:<35} {total:>6}  done (in progress_csv.txt)")
            elif total > max_chunks:
                tprint(f"  {cat:<35} {total:>6}  skipped (>{max_chunks} chunks)")
            elif cat in errored:
                attempts = read_errors().get(cat, {}).get("attempts", "?")
                tprint(f"  {cat:<35} {total:>6}  x errored (attempts: {attempts})")
            else:
                to_dl  = chunks_needing_download(cat, chunks, force=force)
                n_dl   = sum(1 for _, nd in to_dl if nd)
                n_skip = sum(1 for _, nd in to_dl if not nd)
                detail = f"download {n_dl}"
                if n_skip:
                    detail += f", skip {n_skip} (already on disk)"
                tprint(f"  {cat:<35} {total:>6}  {detail}")

        tprint(f"\n  Summary:")
        tprint(f"    Done                   : {len(completed)}")
        tprint(f"    Skipped (>{max_chunks} chunks) : {len(skipped_by_limit)}")
        tprint(f"    Errored                : {len(errored)}")
        tprint(f"    To download            : {len(pending)}")
        if skipped_by_limit:
            tprint(f"\n  Skipped categories (>{max_chunks} chunks):")
            for cat, total in skipped_by_limit:
                tprint(f"    {cat} ({total} chunks)")
        return

    # normal run
    tprint(f"  Total categories : {len(all_categories)}")
    tprint(f"  Already done     : {len(completed)}")
    tprint(f"  Skipped (>{max_chunks} chunks): {len(skipped_by_limit)}")
    tprint(f"  To download      : {len(pending)}")
    tprint()

    if not pending:
        tprint("  nothing to do!")
        return

    succeeded = 0
    failed    = 0

    for i, (category, filenames, total, folder) in enumerate(pending):
        tprint(f"\n{'='*60}")
        tprint(f"  [{i+1}/{len(pending)}] {category}  ({total} chunk(s))")
        tprint(f"{'='*60}")

        attempt  = 0
        ok       = False
        last_err = ""

        while attempt < MAX_RETRIES:
            attempt += 1
            if attempt > 1:
                tprint(f"  retry {attempt}/{MAX_RETRIES} (waiting {RETRY_WAIT}s) ...")
                time.sleep(RETRY_WAIT)

            ok, last_err = download_category(category, folder, filenames, force=force)
            if ok:
                break
            tprint(f"  x attempt {attempt} failed: {last_err}")

        if ok:
            mark_done(category)
            tprint(f"  done: [{category}] complete")
            succeeded += 1
        else:
            log_error(category, last_err)
            tprint(f"  failed: [{category}] after {MAX_RETRIES} attempts - logged to {ERROR_FILE}")
            failed += 1

    # summary
    tprint(f"\n{'='*60}")
    tprint(f"  Download complete")
    tprint(f"  Succeeded : {succeeded}")
    tprint(f"  Failed    : {failed}")
    if failed:
        tprint(f"  See {ERROR_FILE} for details")
    tprint(f"{'='*60}")


if __name__ == "__main__":
    main()
