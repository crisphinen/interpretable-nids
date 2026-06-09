import os, json, subprocess, tempfile, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, DB_PATH, DUCKDB_BIN,
    KNOWN_CLASSES, UNKNOWN_PRIMARY, UNKNOWN_RARE,
    NUMERIC_FEATURES, CAT_FEATURES, ALL_FEATURES,
    SAMPLES_PER_CLASS, VAL_FRAC, TEST_FRAC, RANDOM_SEED,
)

os.makedirs(DATA_DIR, exist_ok=True)

# 1. pull data from DuckDB
ALL_LABEL_CLASSES = KNOWN_CLASSES + UNKNOWN_PRIMARY + UNKNOWN_RARE

# build the IN clause
in_clause = ', '.join(f"'{c}'" for c in ALL_LABEL_CLASSES)

# columns to select: numeric features + raw categoricals + label
select_cols = ', '.join(NUMERIC_FEATURES + CAT_FEATURES + ['detailed_label'])

sql = f"""
COPY (
    SELECT {select_cols}
    FROM preprocessed_sorted
    WHERE detailed_label IN ({in_clause})
) TO '/tmp/hyper_iot_raw.parquet' (FORMAT PARQUET, ROW_GROUP_SIZE 100000);
"""

print("Querying preprocessed_sorted via DuckDB CLI …")
result = subprocess.run(
    [DUCKDB_BIN, DB_PATH],
    input=sql, text=True, capture_output=True
)
if result.returncode != 0:
    print("DuckDB stderr:", result.stderr)
    sys.exit(1)
print("Query done.")

df = pd.read_parquet('/tmp/hyper_iot_raw.parquet')
print(f"Total rows loaded : {len(df):,}")
print("Class distribution:")
print(df['detailed_label'].value_counts().to_string())

# 2. encode categoricals
# Build vocabulary from KNOWN-class rows only (train-distribution).
# 0 is reserved for missing / unseen values.
print("\nBuilding categorical vocabularies from known-class data …")
known_mask = df['detailed_label'].isin(KNOWN_CLASSES)

cat_vocabs = {}
for col in CAT_FEATURES:
    unique_vals = sorted(df.loc[known_mask, col].dropna().unique().tolist())
    cat_vocabs[col] = {v: i + 1 for i, v in enumerate(unique_vals)}  # 1-indexed
    id_col = col + '_id'
    df[id_col] = df[col].map(cat_vocabs[col]).fillna(0).astype(np.float32)
    print(f"  {col}: {len(unique_vals)} unique values")

# 3. numeric: fillna safety net
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(np.float32)

# 4. label encoding
all_labels     = KNOWN_CLASSES + UNKNOWN_PRIMARY + UNKNOWN_RARE
label_to_id    = {l: i for i, l in enumerate(all_labels)}
id_to_label    = {i: l for l, i in label_to_id.items()}
known_ids      = {label_to_id[l] for l in KNOWN_CLASSES}
unk_primary_ids = {label_to_id[l] for l in UNKNOWN_PRIMARY}
unk_rare_ids   = {label_to_id[l] for l in UNKNOWN_RARE
                  if l in label_to_id and df['detailed_label'].isin([l]).any()}

df['label_id'] = df['detailed_label'].map(label_to_id)

# 5. feature matrix columns (final 37)
feat_cols = NUMERIC_FEATURES + [c + '_id' for c in CAT_FEATURES]
assert len(feat_cols) == 37, f"Expected 37 features, got {len(feat_cols)}"

# 6. split known classes
print(f"\nUndersampling known classes to {SAMPLES_PER_CLASS:,} each …")
known_df = df[df['detailed_label'].isin(KNOWN_CLASSES)].copy()

sampled_frames = []
for cls in KNOWN_CLASSES:
    cls_df = known_df[known_df['detailed_label'] == cls]
    n      = min(len(cls_df), SAMPLES_PER_CLASS)
    sampled_frames.append(cls_df.sample(n=n, random_state=RANDOM_SEED))
    print(f"  {cls:20s}: {n:,}")
known_sampled = pd.concat(sampled_frames, ignore_index=True)

# 70 / 15 / 15 stratified split
train_df, temp_df = train_test_split(
    known_sampled, test_size=(VAL_FRAC + TEST_FRAC),
    stratify=known_sampled['label_id'], random_state=RANDOM_SEED
)
val_df, test_known_df = train_test_split(
    temp_df, test_size=0.5,
    stratify=temp_df['label_id'], random_state=RANDOM_SEED
)

print(f"\nSplit sizes:")
print(f"  train      : {len(train_df):,}  ({train_df['detailed_label'].value_counts().to_dict()})")
print(f"  val        : {len(val_df):,}")
print(f"  test_known : {len(test_known_df):,}")

# 7. unknown test sets
test_unk_primary = df[df['detailed_label'].isin(UNKNOWN_PRIMARY)].copy()
test_unk_rare    = df[df['detailed_label'].isin(UNKNOWN_RARE)].copy()
test_unknown_df  = pd.concat([test_unk_primary, test_unk_rare], ignore_index=True)

print(f"  test_unknown (primary) : {len(test_unk_primary):,}")
if len(test_unk_rare):
    print(f"  test_unknown (rare)    : {len(test_unk_rare):,}")
    print(f"  test_unknown (total)   : {len(test_unknown_df):,}")
print()
print("Unknown primary distribution:")
print(test_unk_primary['detailed_label'].value_counts().to_string())
if len(test_unk_rare):
    print("\nUnknown rare distribution:")
    print(test_unk_rare['detailed_label'].value_counts().to_string())

# 8. save to parquet
def save_split(frame, name):
    out = os.path.join(DATA_DIR, f'{name}.parquet')
    cols_to_save = feat_cols + ['label_id', 'detailed_label']
    frame[cols_to_save].to_parquet(out, index=False)
    print(f"Saved {name:15s} → {out}  ({len(frame):,} rows)")

save_split(train_df,      'train')
save_split(val_df,        'val')
save_split(test_known_df, 'test_known')
save_split(test_unknown_df, 'test_unknown')

# 9. save vocab
vocab = {
    'cat_vocabs':       cat_vocabs,
    'label_to_id':      label_to_id,
    'id_to_label':      {str(k): v for k, v in id_to_label.items()},
    'known_classes':    KNOWN_CLASSES,
    'unknown_primary':  UNKNOWN_PRIMARY,
    'unknown_rare':     UNKNOWN_RARE,
    'known_ids':        list(known_ids),
    'unk_primary_ids':  list(unk_primary_ids),
    'unk_rare_ids':     list(unk_rare_ids),
    'feature_cols':     feat_cols,
    'n_features':       len(feat_cols),
    'samples_per_class': SAMPLES_PER_CLASS,
}
vocab_path = os.path.join(DATA_DIR, 'vocab.json')
with open(vocab_path, 'w') as f:
    json.dump(vocab, f, indent=2)
print(f"\nVocab saved → {vocab_path}")
print("\nDone.")
