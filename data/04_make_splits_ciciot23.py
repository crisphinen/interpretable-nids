import os, json, subprocess, tempfile, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cic_config import (
    DATA_DIR, DB_PATH,
    KNOWN_CLASSES, UNKNOWN_PRIMARY, UNKNOWN_RARE,
    NUMERIC_FEATURES, ALL_FEATURES,
    SAMPLES_PER_CLASS, VAL_FRAC, TEST_FRAC, RANDOM_SEED,
)
from config import DUCKDB_BIN

os.makedirs(DATA_DIR, exist_ok=True)

ALL_LABEL_CLASSES = KNOWN_CLASSES + UNKNOWN_PRIMARY + UNKNOWN_RARE
in_clause  = ', '.join(f"'{c}'" for c in ALL_LABEL_CLASSES)
feat_sql   = ', '.join(f'"{c}"' for c in ALL_FEATURES)

# 1. export from DuckDB to parquet via CLI
# Write to DATA_DIR (on work disk), not /tmp (main disk may be full)
raw_parquet = os.path.join(DATA_DIR, '_raw.parquet')
sql = f"""
COPY (
    SELECT {feat_sql}, label
    FROM flows
    WHERE label IN ({in_clause})
) TO '{raw_parquet}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000);
"""

print(f"Querying {DB_PATH} via DuckDB CLI ...")
result = subprocess.run(
    [DUCKDB_BIN, DB_PATH],
    input=sql, text=True, capture_output=True
)
if result.returncode != 0:
    print("DuckDB stderr:", result.stderr)
    sys.exit(1)
print("Query done.")

df = pd.read_parquet(raw_parquet)
os.remove(raw_parquet)
print(f"Loaded {len(df):,} rows across {df['label'].nunique()} classes")
print(df['label'].value_counts().to_string())

# 2. assign integer label IDs
all_cls     = KNOWN_CLASSES + UNKNOWN_PRIMARY + UNKNOWN_RARE
label_to_id = {cls: i for i, cls in enumerate(all_cls)}
df['label_id'] = df['label'].map(label_to_id).astype(np.int32)

# 3. drop NaN / Inf
n_before = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=ALL_FEATURES, inplace=True)
print(f"\nDropped {n_before - len(df):,} rows with NaN/Inf")

# 4. undersample known classes
print("\nUndersampling known classes:")
known_mask = df['label'].isin(KNOWN_CLASSES)
known_df   = df[known_mask]
unknown_df = df[~known_mask]

parts = []
for cls in KNOWN_CLASSES:
    cls_df = known_df[known_df['label'] == cls]
    n      = min(len(cls_df), SAMPLES_PER_CLASS)
    parts.append(cls_df.sample(n=n, random_state=RANDOM_SEED))
    print(f"  {cls:35s}  {len(cls_df):>7,} → {n:,}")
known_balanced = pd.concat(parts, ignore_index=True)

# 5. 70/15/15 stratified split
X_train, X_tmp = train_test_split(
    known_balanced, test_size=VAL_FRAC + TEST_FRAC,
    stratify=known_balanced['label'], random_state=RANDOM_SEED
)
val_frac_of_tmp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
X_val, X_test_k = train_test_split(
    X_tmp, test_size=1 - val_frac_of_tmp,
    stratify=X_tmp['label'], random_state=RANDOM_SEED
)
print(f"\nSplit sizes: train={len(X_train):,}  val={len(X_val):,}  "
      f"test_known={len(X_test_k):,}  test_unknown={len(unknown_df):,}")

# 6. StandardScaler (fit on train only)
print("Fitting StandardScaler on training features ...")
scaler = StandardScaler()
X_train = X_train.copy()
X_val   = X_val.copy()
X_test_k = X_test_k.copy()
unknown_df = unknown_df.copy()

X_train[ALL_FEATURES]  = scaler.fit_transform(X_train[ALL_FEATURES].values)
X_val[ALL_FEATURES]    = scaler.transform(X_val[ALL_FEATURES].values)
X_test_k[ALL_FEATURES] = scaler.transform(X_test_k[ALL_FEATURES].values)
unknown_df[ALL_FEATURES] = scaler.transform(unknown_df[ALL_FEATURES].values)

# 7. save splits
cols_to_save = ALL_FEATURES + ['label_id']
X_train[cols_to_save].to_parquet(   os.path.join(DATA_DIR, 'train.parquet'),        index=False)
X_val[cols_to_save].to_parquet(     os.path.join(DATA_DIR, 'val.parquet'),          index=False)
X_test_k[cols_to_save].to_parquet(  os.path.join(DATA_DIR, 'test_known.parquet'),   index=False)
unknown_df[cols_to_save].to_parquet(os.path.join(DATA_DIR, 'test_unknown.parquet'), index=False)

# 8. save vocab
vocab = {
    'feature_cols':     ALL_FEATURES,
    'label_to_id':      label_to_id,
    'known_classes':    KNOWN_CLASSES,
    'unknown_primary':  UNKNOWN_PRIMARY,
    'unknown_rare':     UNKNOWN_RARE,
    'scaler_mean':      scaler.mean_.tolist(),
    'scaler_scale':     scaler.scale_.tolist(),
}
with open(os.path.join(DATA_DIR, 'vocab.json'), 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"\nSaved splits + vocab.json to: {DATA_DIR}")
print(f"n_features = {len(ALL_FEATURES)}")
