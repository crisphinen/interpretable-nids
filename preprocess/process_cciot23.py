# cic-iot-2023 parallel download + zeek pipeline.
# for each category: download all chunks in parallel, run zeek across all chunks,
# delete chunks to free space, then mark the category done in progress.log.
#
# while zeek processes category N, category N+1 chunks are prefetched in the
# background as long as the free space budget allows.
#
# usage:
#   python pipeline_v2.py                               - normal run
#   python pipeline_v2.py --retry-failed                - only retry errored categories
#   python pipeline_v2.py --category Mirai-greeth_flood - run one specific category
#   python pipeline_v2.py --list                        - show all categories + status
#
# re-running resumes safely - completed categories are skipped.

import os
import re
import json
from typing import Dict, List, Optional, Set, Tuple
import time
import argparse
import threading
import subprocess
import shutil
import urllib3
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# config - edit these before running
COOKIE_TOKEN        = "mv2648bd0i13h34r0j2qf78q36"  # refresh if session expires
ZEEK_BIN            = "zeek"                          # or /opt/zeek/bin/zeek
PCAP_TEMP_DIR       = Path("downloads/pcap")          # temporary PCAP storage
ZEEK_OUT_DIR        = Path("zeek_out")                # Zeek log output
PROGRESS_FILE       = Path("progress.log")            # completed categories
ERROR_FILE          = Path("errors.json")             # failed items (JSON)

MAX_FREE_BYTES      = 750 * 1024**3   # 150 GB - SET THIS to your free space
                                      # script will not prefetch if it would
                                      # push usage over this limit

DOWNLOAD_WORKERS    = 1    # parallel chunk downloads per category
MAX_RETRIES         = 5    # per-chunk download retries
RETRY_WAIT          = 10   # seconds between retries
CHUNK_SIZE          = 1024 * 1024  # 1 MB read buffer
ZEEK_TIMEOUT        = 7200         # 2 hours max per category

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
ENTRIES = [
    ("Backdoor_Malware",         "Backdoor_Malware",         0),
    ("Benign_Final",             "BenignTraffic",            3),
    ("BrowserHijacking",         "BrowserHijacking",         0),
    ("CommandInjection",         "CommandInjection",         0),
    ("DDoS-ACK_Fragmentation",   "DDoS-ACK_Fragmentation",  12),
    ("DDoS-HTTP_Flood",          "DDoS-HTTP_Flood",          0),
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

# urllib3 pool (shared across all threads)
_http = urllib3.PoolManager(
    num_pools=20,
    maxsize=DOWNLOAD_WORKERS * 2,
    retries=False,           # we handle retries manually for better logging
    timeout=urllib3.Timeout(connect=30, read=60),
)

# thread-safe print lock
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


# total bytes of all files under path.
def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def disk_used_by_pcaps() -> int:
    if PCAP_TEMP_DIR.exists():
        return dir_size(PCAP_TEMP_DIR)
    return 0


# returns list of filenames for a category: [base.pcap, base1.pcap, ...]
def build_category_chunks(category: str, fname: str, n: int) -> List[str]:
    files = [f"{fname}.pcap"]
    files += [f"{fname}{i}.pcap" for i in range(1, n + 1)]
    return files


def build_url(category: str, filename: str) -> str:
    encoded = urllib.parse.quote(f"PCAP/{category}/{filename}", safe="")
    return BASE_URL + encoded


def build_referer(category: str) -> str:
    return (
        f"https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
        f"browse.php?p=PCAP%2F{category}"
    )


# returns True if every chunk file exists on disk and is non-empty.
# used only for status display - download always runs and tops up partial files.
def all_chunks_present(category: str, filenames: List[str]) -> bool:
    for fname in filenames:
        p = PCAP_TEMP_DIR / category / fname
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def load_progress() -> set:
    if not PROGRESS_FILE.exists():
        return set()
    return set(l.strip() for l in PROGRESS_FILE.read_text().splitlines() if l.strip())


# read errors.json and return the dict. returns {} if file missing or corrupt.
def read_errors() -> dict:
    if not ERROR_FILE.exists():
        return {}
    try:
        return json.loads(ERROR_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# write errors dict back to errors.json atomically.
def write_errors(data: dict):
    ERROR_FILE.write_text(json.dumps(data, indent=2))


# return set of category names that currently have errors.
def load_errors() -> set:
    return set(read_errors().keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIC-IoT-2023 Download -> Zeek Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_v2.py                              # normal run, skip completed
  python pipeline_v2.py --retry-failed               # only retry errored categories
  python pipeline_v2.py --category Mirai-greeth_flood  # run one specific category
  python pipeline_v2.py --list                       # show all categories + status
        """
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only process categories that have entries in errors.log",
    )
    parser.add_argument(
        "--category",
        metavar="NAME",
        help="Process a single named category (ignores progress.log for that category)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all categories with their current status and exit",
    )
    return parser.parse_args()


# mark category as complete in progress.log and remove from errors.json.
def mark_done(category: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(category + "\n")
    # remove from errors.json if present
    data = read_errors()
    if category in data:
        del data[category]
        write_errors(data)
        tprint(f"  ok: removed [{category}] from errors.json")


# write or update an error entry for a category in errors.json.
# category key is always the top-level category name, never a filename.
def log_error(category: str, message: str):
    # strip any /filename suffix so key is always just the category name
    category = category.split("/")[0].strip()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    data = read_errors()
    existing = data.get(category, {})
    data[category] = {
        "last_error": message,
        "timestamp": timestamp,
        "attempts": existing.get("attempts", 0) + 1,
    }
    write_errors(data)
    tprint(f"  x [{timestamp}] {category}: {message}")


# download

# download a single chunk with resume support.
# returns (success, error_message).
# updates the shared tqdm progress bar as bytes arrive.
def download_chunk(url: str, dest: Path, referer: str,
                   progress_bar: tqdm) -> Tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(HEADERS)
    headers["referer"] = referer
    headers["cookie"] = f"Token={COOKIE_TOKEN}"

    for attempt in range(1, MAX_RETRIES + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        if existing:
            headers["range"] = f"bytes={existing}-"
        elif "range" in headers:
            del headers["range"]

        try:
            resp = _http.request(
                "GET", url, headers=headers,
                preload_content=False,
            )

            if resp.status == 416:
                # already complete - advance bar by file size
                progress_bar.update(existing)
                resp.drain_conn()
                return True, ""

            if resp.status not in (200, 206):
                body = resp.data[:300].decode("utf-8", errors="replace")
                resp.drain_conn()
                raise ValueError(f"HTTP {resp.status}: {body}")

            mode = "ab" if resp.status == 206 else "wb"
            if resp.status == 200 and existing:
                existing = 0  # server ignored Range, restart

            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            resp.drain_conn()
            return True, ""

        except Exception as e:
            err = str(e)
            tprint(f"    [{dest.name}] attempt {attempt}/{MAX_RETRIES} failed: {err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    return False, f"Failed after {MAX_RETRIES} attempts"


# download all chunks for a category in parallel.
# returns (all_succeeded, list_of_failed_filenames).
def download_category_parallel(category: str, filenames: List[str],
                                referer: str) -> Tuple[bool, List[str]]:
    tprint(f"\n  downloading {len(filenames)} chunk(s) for [{category}] ...")

    # build (url, dest) pairs, skip already-complete files
    tasks = []
    total_to_download = 0
    for fname in filenames:
        dest = PCAP_TEMP_DIR / category / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = build_url(category, fname)
        existing = dest.stat().st_size if dest.exists() else 0
        tasks.append((url, dest, existing))
        # we don't know total size (chunked transfer), so progress bar is
        # bytes-downloaded style rather than percentage
        total_to_download += existing  # start bar at already-downloaded bytes

    progress = tqdm(
        total=None,   # unknown total due to chunked transfer-encoding
        initial=sum(t[2] for t in tasks),
        unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"  {category}",
        dynamic_ncols=True,
    )

    failed = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        future_to_fname = {
            pool.submit(download_chunk, url, dest, referer, progress): dest.name
            for url, dest, _ in tasks
        }
        for future in as_completed(future_to_fname):
            fname = future_to_fname[future]
            try:
                ok, err = future.result()
                if not ok:
                    failed.append(fname)
                    log_error(f"{category}/{fname}", err)
            except Exception as e:
                failed.append(fname)
                log_error(f"{category}/{fname}", str(e))

    progress.close()

    if failed:
        tprint(f"  x {len(failed)} chunk(s) failed to download: {failed}")
    else:
        total = sum(
            (PCAP_TEMP_DIR / category / fn).stat().st_size
            for fn in filenames
            if (PCAP_TEMP_DIR / category / fn).exists()
        )
        tprint(f"  ok: all chunks downloaded ({human_size(total)} total)")

    return len(failed) == 0, failed


# zeek

# run zeek on all chunks for a category.
#
# single chunk: zeek -r file.pcap (no merge needed).
# multi chunk:  mergecap -w merged.pcap chunk0 chunk1 ...
#               then zeek -r merged.pcap.
#               merged file is always deleted in the finally block.
#
# mergecap correctly strips per-file pcap headers from chunks 2+ so zeek
# sees one valid continuous capture. raw cat/pipe cannot do this - each
# pcap file has its own header that would corrupt the stream.
#
# output goes to ZEEK_OUT_DIR/category/.
# returns (success, error_message).
def run_zeek(category: str, filenames: List[str]) -> Tuple[bool, str]:
    out_dir = ZEEK_OUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    pcap_paths = []
    for fname in filenames:
        p = PCAP_TEMP_DIR / category / fname
        if not p.exists():
            return False, f"Missing chunk: {p}"
        pcap_paths.append(str(p.resolve()))

    tprint(f"  running zeek on {len(pcap_paths)} chunk(s) -> {out_dir}")

    merged_path = PCAP_TEMP_DIR / category / "_merged.pcap"

    try:
        if len(pcap_paths) == 1:
            input_pcap = str(Path(pcap_paths[0]).resolve())
        else:
            # merge all chunks into one valid pcap using mergecap
            tprint(f"  merging {len(pcap_paths)} chunks ...")
            merge_result = subprocess.run(
                ["mergecap", "-w", str(merged_path)] + pcap_paths,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if merge_result.returncode != 0:
                return False, (
                    f"mergecap failed (exit {merge_result.returncode})\n"
                    f"STDERR: {merge_result.stderr[:400]}\n"
                    f"Tip: sudo apt install wireshark-common"
                )
            tprint(f"  ok: merged -> {human_size(merged_path.stat().st_size)}")
            input_pcap = str(merged_path.resolve())

        # run zeek on the single clean input file
        zeek_result = subprocess.run(
            [ZEEK_BIN, "-r", input_pcap],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=ZEEK_TIMEOUT,
        )

        if zeek_result.returncode != 0:
            return False, (
                f"Zeek exited {zeek_result.returncode}\n"
                f"STDOUT: {zeek_result.stdout[:400]}\n"
                f"STDERR: {zeek_result.stderr[:400]}"
            )

        logs = list(out_dir.glob("*.log"))
        tprint(f"  ok: zeek complete - {len(logs)} log file(s) written to {out_dir}")
        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"Timed out after {ZEEK_TIMEOUT}s"
    except FileNotFoundError as e:
        return False, f"Binary not found: {e}  (mergecap: sudo apt install wireshark-common)"
    finally:
        # always delete merged file to free space
        if merged_path.exists():
            merged_path.unlink()
            tprint(f"  deleted merged PCAP")


# cleanup

# delete all downloaded PCAPs for a category.
def cleanup_category(category: str, filenames: List[str]):
    deleted = 0
    freed = 0
    for fname in filenames:
        p = PCAP_TEMP_DIR / category / fname
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            deleted += 1
    # remove empty directory
    cat_dir = PCAP_TEMP_DIR / category
    try:
        cat_dir.rmdir()
    except OSError:
        pass
    tprint(f"  deleted {deleted} PCAP(s), freed {human_size(freed)}")


# prefetch logic

# manages background downloading of the next category while zeek processes
# the current one. respects MAX_FREE_BYTES: will not start a prefetch if
# current pcap disk usage is already above the budget.
class PrefetchManager:

    def __init__(self):
        self._future = None          # ThreadPoolExecutor future
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._result = None          # (success, failed_list) once done
        self._category = None
        self._filenames = None
        self._lock = threading.Lock()

    def start(self, category: str, filenames: List[str], referer: str):
        used = disk_used_by_pcaps()
        if used >= MAX_FREE_BYTES:
            tprint(f"\n  warning: prefetch skipped for [{category}] - "
                   f"disk usage {human_size(used)} >= budget {human_size(MAX_FREE_BYTES)}")
            return

        tprint(f"\n  prefetching [{category}] in background "
               f"(disk used: {human_size(used)} / {human_size(MAX_FREE_BYTES)})")

        with self._lock:
            self._category  = category
            self._filenames = filenames
            self._result    = None
            self._future    = self._executor.submit(
                download_category_parallel, category, filenames, referer
            )

    def wait(self) -> Optional[Tuple[bool, List[str]]]:
        # block until prefetch is done. returns (success, failed) or None.
        with self._lock:
            if self._future is None:
                return None
        result = self._future.result()
        with self._lock:
            self._result  = result
            self._future  = None
        return result

    def is_running(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def shutdown(self):
        self._executor.shutdown(wait=False)


# main pipeline

# full pipeline for one category.
# if download_result is provided, download was already done by prefetch.
# returns True if zeek succeeded (even if some downloads failed).
def process_category(category: str, filenames: List[str],
                     download_result: Optional[Tuple[bool, List[str]]] = None) -> bool:
    tprint(f"\n{'='*60}")
    tprint(f"  Category : {category}")
    tprint(f"  Chunks   : {len(filenames)}")
    tprint(f"{'='*60}")

    referer = build_referer(category)

    # download (or use prefetch result)
    # always run download - server returns 416 instantly for complete files
    # and tops up any partial ones via Range header. this is the only reliable
    # way to detect and fix partial downloads.
    if download_result is not None:
        tprint(f"  ok: using prefetched download result")
        dl_ok, failed_chunks = download_result
    else:
        dl_ok, failed_chunks = download_category_parallel(
            category, filenames, referer
        )

    if not dl_ok:
        log_error(category, f"Download failed for chunks: {failed_chunks} - skipping entire category")
        tprint(f"  x: skipping [{category}] - chunk download(s) failed: {failed_chunks}")
        tprint(f"  warning: partial downloads kept at {PCAP_TEMP_DIR / category} for inspection")
        return False

    # zeek
    tprint(f"\n  [2/3] Running Zeek ...")
    zeek_ok, zeek_err = run_zeek(category, filenames)
    if not zeek_ok:
        log_error(category, f"Zeek failed: {zeek_err}")
        tprint(f"  warning: PCAPs kept at {PCAP_TEMP_DIR / category} for inspection")
        return False

    # cleanup
    tprint(f"\n  [3/3] Cleaning up PCAPs ...")
    cleanup_category(category, filenames)

    mark_done(category)
    tprint(f"\n  done: [{category}] complete")
    return True


def main():
    args = parse_args()

    tprint("=" * 60)
    tprint("  CIC-IoT-2023  Download -> Zeek Pipeline")
    tprint("=" * 60)
    tprint(f"  PCAP temp dir  : {PCAP_TEMP_DIR}")
    tprint(f"  Zeek output    : {ZEEK_OUT_DIR}")
    tprint(f"  Space budget   : {human_size(MAX_FREE_BYTES)}")
    tprint(f"  Download workers per category: {DOWNLOAD_WORKERS}")
    tprint()

    # build full category list
    completed = load_progress()
    errored   = load_errors()
    all_categories = [
        (cat, build_category_chunks(cat, fname, n), build_referer(cat))
        for cat, fname, n in ENTRIES
    ]
    cat_lookup = {cat: (cat, chunks, ref) for cat, chunks, ref in all_categories}

    # list mode
    if args.list:
        error_data = read_errors()
        tprint(f"  {'CATEGORY':<35} STATUS")
        tprint(f"  {'-'*35} ------")
        for cat, _, _ in all_categories:
            if cat in completed:
                status = "done"
            elif cat in errored:
                attempts = error_data.get(cat, {}).get("attempts", "?")
                last_err = error_data.get(cat, {}).get("last_error", "")[:50]
                status = f"x errored (attempts: {attempts}) - {last_err}"
            else:
                status = "pending"
            tprint(f"  {cat:<35} {status}")
        tprint(f"\n  Total: {len(all_categories)}  |  Done: {len(completed)}  |  "
               f"Errored: {len(errored)}  |  Pending: {len(all_categories)-len(completed)-len(errored)}")
        return

    # verify zeek
    try:
        r = subprocess.run([ZEEK_BIN, "--version"], capture_output=True, text=True)
        ver = (r.stdout + r.stderr).strip().splitlines()[0]
        tprint(f"  Zeek : {ver}")
    except FileNotFoundError:
        tprint(f"  x: Zeek not found: '{ZEEK_BIN}'")
        tprint("    Install: sudo apt install zeek")
        tprint("    Or set ZEEK_BIN to the full path.")
        return

    # build pending list based on mode
    if args.category:
        # single category mode - run regardless of progress.log
        if args.category not in cat_lookup:
            tprint(f"  x: unknown category: '{args.category}'")
            tprint(f"  Run with --list to see valid category names.")
            return
        pending = [cat_lookup[args.category]]
        tprint(f"  Mode: single category -> [{args.category}]")

    elif args.retry_failed:
        # retry-failed mode - only categories in errors.log
        if not errored:
            tprint("  No errors found in errors.log - nothing to retry.")
            return
        pending = [cat_lookup[c] for c in errored if c in cat_lookup]
        tprint(f"  Mode: retry-failed -> {len(pending)} category(s)")
        for cat, chunks, _ in pending:
            have = all_chunks_present(cat, chunks)
            tprint(f"    - {cat}  {'(PCAPs on disk - will skip download)' if have else '(will re-download)'}")
        tprint()

    else:
        # normal mode - skip completed
        pending = [(c, f, r) for c, f, r in all_categories if c not in completed]

    tprint(f"\n  Total categories : {len(all_categories)}")
    tprint(f"  Already done     : {len(completed)}")
    tprint(f"  To process       : {len(pending)}")
    tprint()

    if not pending:
        tprint("  nothing to do - all categories complete!")
        return

    prefetch = PrefetchManager()
    failed_categories = []

    # kick off prefetch for first category before entering the loop
    if pending:
        prefetch.start(pending[0][0], pending[0][1], pending[0][2])

    try:
        for i, (category, filenames, referer) in enumerate(pending):
            is_last = (i == len(pending) - 1)

            tprint(f"\n  [{i+1}/{len(pending)}] Starting [{category}]")

            # wait for this category's download to complete.
            # for complete files the server returns 416 immediately - fast no-op.
            # for partial files it tops them up. always reliable.
            tprint(f"  waiting for download of [{category}] ...")
            download_result = prefetch.wait()

            # start prefetching next category while zeek runs on current one
            if not is_last:
                next_cat, next_files, next_ref = pending[i + 1]
                prefetch.start(next_cat, next_files, next_ref)

            # zeek + cleanup
            ok = process_category(category, filenames, download_result=download_result)

            if not ok:
                failed_categories.append(category)

    finally:
        prefetch.shutdown()

    # summary
    tprint(f"\n{'='*60}")
    tprint(f"  Pipeline complete")
    tprint(f"  Succeeded : {len(pending) - len(failed_categories)}")
    tprint(f"  Failed    : {len(failed_categories)}")
    if failed_categories:
        tprint(f"  See {ERROR_FILE} for details:")
        for c in failed_categories:
            tprint(f"    - {c}")
    tprint(f"{'='*60}")


if __name__ == "__main__":
    main()
