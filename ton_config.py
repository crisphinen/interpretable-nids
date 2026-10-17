"""TON_IoT config: network subset (Zeek conn + protocol logs), 5 known / 5 held-out families.

50,000 flows per known class and an unknown pool of 373,857 flows (each held-out family capped at
100,000: 3 x 100,000 + ransomware 72,805 + MITM 1,052). 23 features: the 8 numeric conn-log fields
and the numeric/flag fields of the dns and http logs (18 in total, z-scored), and 5 categorical
fields integer-encoded from known training rows (not scaled).
Addresses, ports, timestamps and free-text fields are excluded.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "ton")
CSV_GLOB = os.environ.get("TON_CSV_GLOB", os.path.join(ROOT, "downloads", "TON_IoT", "Processed_Network_dataset",
                                                   "Network_dataset_*.csv"))

KNOWN_CLASSES = ["normal", "scanning", "ddos", "password", "injection"]
UNKNOWN_CLASSES = ["dos", "xss", "backdoor", "ransomware", "mitm"]

CONN_NUMERIC = ["duration", "src_bytes", "dst_bytes", "missed_bytes",
                "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
DNS_NUMERIC = ["dns_qclass", "dns_qtype", "dns_rcode"]
DNS_FLAGS = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected"]          # T/F/- -> 1/0/0
HTTP_NUMERIC = ["http_request_body_len", "http_response_body_len", "http_status_code"]
NUMERIC_FEATURES = CONN_NUMERIC + DNS_NUMERIC + DNS_FLAGS + HTTP_NUMERIC   # 18, z-scored
CAT_FEATURES = ["proto", "service", "conn_state", "ssl_version", "http_method"]   # 5
# raw copies kept for the concept rules, which use raw Zeek units
CONCEPT_RAW = ["duration", "src_pkts", "dst_pkts", "conn_state", "service"]

SAMPLES_PER_CLASS = 50_000
UNKNOWN_CAP = 100_000
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
