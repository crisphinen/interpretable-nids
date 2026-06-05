"""CIC-IoT-2023 config: 39 DPKT features, 5 known / 29 unknown classes."""
import os

# paths
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, 'data', 'cic')
RESULTS_DIR = os.path.join(ROOT, 'results', 'cic')
DB_PATH     = os.path.join(ROOT, 'data', 'ciciot23.duckdb')

# class definitions
# Known: Benign + one representative from each broad attack family
# (chosen for diversity and class balance ≥ 50K rows)
KNOWN_CLASSES = [
    'Benign_Final',
    'DDoS-SYN_Flood',
    'Mirai-greeth_flood',
    'Recon-PortScan',
    'VulnerabilityScan',
]

# Primary unknowns: large enough for per-class reporting
UNKNOWN_PRIMARY = [
    'DDoS-UDP_Flood',
    'DDoS-TCP_Flood',
    'DDoS-ICMP_Flood',
    'DDoS-RSTFINFlood',
    'DDoS-PSHACK_Flood',
    'DDoS-SynonymousIP_Flood',
    'DDoS-ACK_Fragmentation',
    'DDoS-UDP_Fragmentation',
    'DDoS-HTTP_Flood',
    'DDoS-SlowLoris',
    'DDoS-ICMP_Fragmentation',
    'DoS-SYN_Flood',
    'DoS-UDP_Flood',
    'DoS-TCP_Flood',
    'DoS-HTTP_Flood',
    'Mirai-greip_flood',
    'Mirai-udpplain',
    'Recon-HostDiscovery',
    'Recon-OSScan',
    'Recon-PingSweep',
    'MITM-ArpSpoofing',
    'DNS_Spoofing',
    'DictionaryBruteForce',
]

# Rare unknowns: small classes pooled for reporting
UNKNOWN_RARE = [
    'BrowserHijacking',
    'CommandInjection',
    'XSS',
    'SqlInjection',
    'Backdoor_Malware',
    'Uploading_Attack',
]

# data splits
SAMPLES_PER_CLASS = 50_000   # Recon-PortScan has 82K; all classes balanced to 50K
VAL_FRAC          = 0.15
TEST_FRAC         = 0.15
RANDOM_SEED       = 42

# features: 39 DPKT columns (all numeric, no Zeek categoricals)
NUMERIC_FEATURES = [
    'header_length', 'protocol_type', 'time_to_live', 'rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'http', 'https', 'dns', 'telnet', 'smtp', 'ssh', 'irc',
    'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'igmp', 'ipv', 'llc',
    'tot_sum', 'min', 'max', 'avg', 'std', 'tot_size',
    'iat', 'number', 'variance',
]
CAT_FEATURES = []          # none for CIC-IoT-2023 (DPKT has no flow-state fields)
ALL_FEATURES = NUMERIC_FEATURES
N_FEATURES   = len(ALL_FEATURES)   # 39

# model
EMBED_DIM   = 64
CURVATURE   = 1.0
HIDDEN_DIM  = 256
TEMPERATURE = 0.1

# training
EPOCHS       = 50
BATCH_SIZE   = 512
LR_MLP       = 1e-3
LR_PROTO     = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10

# evaluation
FPR_LEVELS = [0.01, 0.05, 0.10]
