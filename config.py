"""Central config — change values here, everything else reads from here."""
import os

# paths
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
DB_PATH     = '/home/Ngari/Research/iot23_3.duckdb'
DUCKDB_BIN  = '/home/Ngari/Research/duckdb'

# class definitions
KNOWN_CLASSES   = ['Benign', 'C&C-HeartBeat', 'DDoS', 'Okiru']
UNKNOWN_PRIMARY = ['C&C', 'Attack', 'C&C-HeartBeat-Attack']
UNKNOWN_RARE    = ['C&C-FileDownload', 'C&C-Torii', 'C&C-Mirai',
                   'C&C-HeartBeat-FileDownload', 'FileDownload', 'Okiru-Attack']

# data splits
SAMPLES_PER_CLASS = 23_000   # matches C&C-HeartBeat ceiling; all classes balanced
VAL_FRAC          = 0.15
TEST_FRAC         = 0.15
RANDOM_SEED       = 42

# features (RF-selected 99% cumulative importance, from windows/vocabs.json)
NUMERIC_FEATURES = [
    'Telnet', 'orig_byte_rate', 'resp_ip_bytes', 'Number', 'Tot_sum',
    'Srate', 'IAT', 'HTTPS', 'resp_pkts', 'orig_pkts', 'Rate',
    'orig_pkt_rate', 'Max', 'AVG', 'resp_bytes', 'ack_count',
    'orig_ip_bytes', 'Magnitude', 'Variance', 'Std', 'direction',
    'syn_count', 'Radius', 'duration', 'byte_ratio', 'IRC', 'orig_bytes',
    'Min', 'SSH', 'fin_count', 'rst_count', 'HTTP', 'fin_flag_number',
    'rst_flag_number',
]
CAT_FEATURES = ['conn_state', 'history', 'service']   # raw → integer-encoded
ALL_FEATURES = NUMERIC_FEATURES + ['conn_state_id', 'history_id', 'service_id']
N_FEATURES   = len(ALL_FEATURES)   # 37

# model
EMBED_DIM  = 64
CURVATURE  = 1.0      # Poincaré ball curvature c; c→0 approaches Euclidean
HIDDEN_DIM = 256
TEMPERATURE = 0.1     # softmax temperature on distances

# training
EPOCHS      = 50
BATCH_SIZE  = 512
LR_MLP      = 1e-3    # standard Adam for MLP weights
LR_PROTO    = 1e-3    # RiemannianAdam for manifold prototypes
WEIGHT_DECAY = 1e-4
PATIENCE    = 10      # early stopping on val weighted-F1

# evaluation
FPR_LEVELS  = [0.01, 0.05, 0.10]   # detection rate reported at these FPR levels
