# cic-iot-2023 download-only script for pcap files.
# downloads all pcap files for each category. no zeek, no pipeline.
#
# completeness logic for categories not in progress2.txt:
#   every non-last chunk must be >= 2 gb (server splits at 2 gb).
#   the last chunk is always re-downloaded (cannot verify completeness).
#   any non-last chunk under 2 gb is re-downloaded (was truncated).
#   single-chunk categories are always re-downloaded.
#
# categories with more chunks than MAX_CHUNKS are skipped silently.
#
# usage:
#   python downloader.py                          - download all eligible categories
#   python downloader.py --retry-failed           - retry only categories in errors.json
#   python downloader.py --dry-run                - show what would be downloaded
#   python downloader.py --max-chunks 10          - override chunk limit (default 5)
#   python downloader.py --category Benign_Final  - single category

import json
import argparse
import threading
import time
import urllib3
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# config - edit before running
COOKIE_TOKEN    = "nqterct6q3o3ho944qq0ugpdpq"   # refresh if session expires
PCAP_DIR        = Path("downloads/pcap")          # where PCAPs are saved
PROGRESS_FILE   = Path("progress2.txt")           # completed categories
ERROR_FILE      = Path("errors.json")             # failed categories (JSON)

MAX_CHUNKS      = 5        # skip categories with more chunks than this
MAX_RETRIES     = 5        # attempts per category before giving up
RETRY_WAIT      = 10       # seconds between retries
#CHUNK_SIZE      = 2 * 1024**3   # 2 GB - non-last chunks must be at least this
CHUNK_SIZE = 2_000_000_000 # a little below 2GB. non last chunks must be at least this
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

# All dataset entries: (category, base_filename, n_chunks)
# n_chunks = number of EXTRA chunks after the base file
# e.g. n_chunks=3 means: base.pcap, base1.pcap, base2.pcap, base3.pcap (4 files total)
ENTRIES = [
    ("Backdoor_Malware",         "Backdoor_Malware",         0),
    ("Benign_Final",             "BenignTraffic",            3),
    ("BrowserHijacking",         "BrowserHijacking",         0),
    ("CommandInjection",         "CommandInjection",         0),
    ("DDoS-ACK_Fragmentation",   "DDoS-ACK_Fragmentation",  12),
    ("DDoS-HTTP_Flood",          "DDoS-HTTP_Flood-",          0),
    ("DDoS-ICMP_Flood",          "DDoS-ICMP_Flood",         26),
    ("DDoS-ICMP_Fragmentation",  "DDoS-ICMP_Fragmentation", 19),
    ("DDoS-PSHACK_Flood",        "DDoS-PSHACK_Flood",       15),
    ("DDoS-RSTFINFlood",         "DDoS-RSTFINFlood",        15),
    ("DDoS-SYN_Flood",           "DDoS-SYN_Flood",          15),
    ("DDoS-SlowLoris",           "DDoS-SlowLoris",           0),
    ("DDoS-SynonymousIP_Flood",  "DDoS-SynonymousIP_Flood", 13),
    ("DDoS-TCP_Flood",           "DDoS-TCP_Flood",          17),
    ("DDoS-UDP_Flood",           "DDoS-UDP_Flood",          20),
    ("DDoS-UDP_Fragmentation",   "DDoS-UDP_Fragmentation",  12),
    ("DNS_Spoofing",             "DNS_Spoofing",              0),
    ("DictionaryBruteForce",     "DictionaryBruteForce",     0),
    ("DoS-HTTP_Flood",           "DoS-HTTP_Flood",            1),
    ("DoS-SYN_Flood",            "DoS-SYN_Flood",             7),
    ("DoS-TCP_Flood",            "DoS-TCP_Flood",            10),
    ("DoS-UDP_Flood",            "DoS-UDP_Flood",            16),
    ("MITM-ArpSpoofing",         "MITM-ArpSpoofing",          1),
    ("Mirai-greeth_flood",       "Mirai-greeth_flood",       28),
    ("Mirai-greip_flood",        "Mirai-greip_flood",        21),
    ("Mirai-udpplain",           "Mirai-udpplain",           24),
    ("Recon-HostDiscovery",      "Recon-HostDiscovery",       0),
    ("Recon-OSScan",             "Recon-OSScan",              0),
    ("Recon-PingSweep",          "Recon-PingSweep",           0),
    ("Recon-PortScan",           "Recon-PortScan",            0),
    ("SqlInjection",             "SqlInjection",              0),
    ("Uploading_Attack",         "Uploading_Attack",          0),
    ("VulnerabilityScan",        "VulnerabilityScan",         0),
    ("XSS",                      "XSS",                       0),
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

def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# returns list of chunk filenames: [base.pcap, base1.pcap, ..., baseN.pcap]
def build_chunks(fname: str, n: int) -> List[str]:
    files = [f"{fname}.pcap"]
    files += [f"{fname}{i}.pcap" for i in range(1, n + 1)]
    return files


def build_url(category: str, filename: str) -> str:
    encoded = urllib.parse.quote(f"PCAP/{category}/{filename}", safe="")
    return BASE_URL + encoded


def build_referer(category: str) -> str:
    encoded = urllib.parse.quote(f"PCAP/{category}/", safe="")
    return f"https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/download.php?file={encoded}"


# progress and error tracking

def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    return set(l.strip() for l in PROGRESS_FILE.read_text().splitlines() if l.strip())


def mark_done(category: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(category + "\n")
    # remove from errors if it was there
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

# for a category NOT in progress2.txt, determine which chunks need downloading.
# rules:
#   single chunk (1 file total): always re-download.
#   multi chunk non-last: re-download if missing or size < CHUNK_SIZE (2 gb).
#   multi chunk last: always re-download.
# returns list of (filename, needs_download) for all chunks.
def chunks_needing_download(category: str, filenames: List[str]) -> List[Tuple[str, bool]]:
    result = []
    n = len(filenames)

    for i, fname in enumerate(filenames):
        is_last = (i == n - 1)
        p = PCAP_DIR / category / fname

        if n == 1:
            # single chunk - always re-download
            result.append((fname, True))
        elif is_last:
            # last chunk - always re-download
            result.append((fname, True))
        else:
            # non-last chunk - re-download if missing or under 2 gb
            if not p.exists() or p.stat().st_size < CHUNK_SIZE:
                result.append((fname, True))
            else:
                result.append((fname, False))

    return result


# download

# download a single file from url to dest.
# always overwrites (server doesn't support range requests).
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
# downloads one chunk at a time (server throttles).
# returns (success, error_message).
def download_category(category: str, filenames: List[str]) -> Tuple[bool, str]:
    referer = build_referer(category)
    to_download = chunks_needing_download(category, filenames)

    needs = [(f, nd) for f, nd in to_download if nd]
    skipped = [(f, nd) for f, nd in to_download if not nd]

    if skipped:
        tprint(f"  ok: skipping {len(skipped)} chunk(s) already complete")

    if not needs:
        tprint(f"  ok: all chunks already complete - nothing to download")
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
            url = build_url(category, fname)
            dest = PCAP_DIR / category / fname

            # delete partial file before fresh download
            if dest.exists():
                dest.unlink()

            ok, err = download_file(url, dest, referer, pbar)
            if not ok:
                return False, f"Failed to download [{fname}]: {err}"

    tprint(f"  ok: all chunks downloaded for [{category}]")
    return True, ""


# main

def parse_args():
    p = argparse.ArgumentParser(description="CIC-IoT-2023 PCAP Downloader")
    p.add_argument("--retry-failed", action="store_true",
                   help="Only retry categories currently in errors.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded without downloading anything")
    p.add_argument("--category", metavar="NAME",
                   help="Download a single specific category")
    p.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                   help=f"Skip categories with more chunks than this (default: {MAX_CHUNKS})")
    return p.parse_args()


def main():
    args = parse_args()
    max_chunks = args.max_chunks

    tprint("=" * 60)
    tprint("  CIC-IoT-2023  PCAP Downloader")
    tprint("=" * 60)
    tprint(f"  PCAP dir       : {PCAP_DIR}")
    tprint(f"  Progress file  : {PROGRESS_FILE}")
    tprint(f"  Max chunks     : {max_chunks}")
    tprint()

    # build full category list with chunk filenames
    all_categories = [
        (cat, build_chunks(fname, n), n + 1)  # total_chunks = n+1
        for cat, fname, n in ENTRIES
    ]
    cat_lookup = {cat: (cat, chunks, total) for cat, chunks, total in all_categories}

    completed = load_progress()
    errored   = load_errors()

    # build pending list
    if args.category:
        if args.category not in cat_lookup:
            tprint(f"  error: unknown category: '{args.category}'")
            tprint(f"  valid categories:")
            for cat, _, _ in all_categories:
                tprint(f"    {cat}")
            return
        pending = [cat_lookup[args.category]]

    elif args.retry_failed:
        if not errored:
            tprint("  ok: no failed categories in errors.json")
            return
        pending = [cat_lookup[cat] for cat in errored if cat in cat_lookup]
        tprint(f"  retrying {len(pending)} failed category/categories")

    else:
        pending = [
            (cat, chunks, total)
            for cat, chunks, total in all_categories
            if cat not in completed
        ]

    # apply chunk limit filter
    skipped_by_limit = [(cat, total) for cat, chunks, total in pending if total > max_chunks]
    pending = [(cat, chunks, total) for cat, chunks, total in pending if total <= max_chunks]

    # dry run
    if args.dry_run:
        tprint(f"  DRY RUN - nothing will be downloaded\n")
        tprint(f"  {'CATEGORY':<35} {'CHUNKS':>6}  ACTION")
        tprint(f"  {'-'*35} {'-'*6}  ------")

        for cat, chunks, total in all_categories:
            if cat in completed:
                tprint(f"  {cat:<35} {total:>6}  done (in progress2.txt)")
            elif total > max_chunks:
                tprint(f"  {cat:<35} {total:>6}  skipped (>{max_chunks} chunks)")
            elif cat in errored:
                attempts = read_errors().get(cat, {}).get("attempts", "?")
                tprint(f"  {cat:<35} {total:>6}  x errored (attempts: {attempts})")
            else:
                # show which chunks would actually be downloaded
                to_dl = chunks_needing_download(cat, chunks)
                n_dl = sum(1 for _, nd in to_dl if nd)
                n_skip = sum(1 for _, nd in to_dl if not nd)
                detail = f"download {n_dl}"
                if n_skip:
                    detail += f", skip {n_skip} (already complete)"
                tprint(f"  {cat:<35} {total:>6}  {detail}")

        tprint(f"\n  Summary:")
        tprint(f"    Done          : {len(completed)}")
        tprint(f"    Skipped (>{max_chunks} chunks): {len(skipped_by_limit)}")
        tprint(f"    Errored       : {len(errored)}")
        tprint(f"    To download   : {len(pending)}")
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
    failed = 0

    for i, (category, filenames, total) in enumerate(pending):
        tprint(f"\n{'='*60}")
        tprint(f"  [{i+1}/{len(pending)}] {category}  ({total} chunk(s))")
        tprint(f"{'='*60}")

        attempt = 0
        ok = False
        last_err = ""

        while attempt < MAX_RETRIES:
            attempt += 1
            if attempt > 1:
                tprint(f"  retry {attempt}/{MAX_RETRIES} (waiting {RETRY_WAIT}s) ...")
                time.sleep(RETRY_WAIT)

            ok, last_err = download_category(category, filenames)
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
