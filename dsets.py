# dataset registry shared by cbm/ and nesy/.
# ctu   - the original split (data/), 37 features selected on all CTU labels (lambda ablations)
# ctuk  - main CTU split (data/ctuk/), features selected on known-class training rows only
# ctuko - the same rows as ctuk with the submitted 37-feature list (paired control)
# cic   - CIC-IoT-2023 (data/cic/)
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CTU_DIRS = {"ctu": "data", "ctuk": "data/ctuk", "ctuko": "data/ctuko"}
# robustness splits (06_make_kfs_splits.py --split alt|temporal): own re-selection, and _o = the
# main split's 31-feature list (ctuk) on the same rows
CTU_DIRS.update({n: f"data/{n}" for n in ("ctualt", "ctualt_o", "ctut", "ctut_o")})
# device-disjoint splits (06_make_kfs_splits.py --split device): every known-class flow of the held-out
# host is test-only; _o = the main split's 31-feature list on the same rows
CTU_DIRS.update({n: f"data/{n}" for n in ("ctud_h198", "ctud_o_h198", "ctud_h111", "ctud_o_h111",
                                         "ctud_h108", "ctud_o_h108")})
# restored-features control (08_make_ctuk_n.py): ctuk rows, MLP/CBM input = NeSy-NIDS's 33 columns
CTU_DIRS.update({"ctuk_n": "data/ctuk_n"})
# leave-family-out folds (cbm/run_lofo.py --make-folds): vocab variants over the base split's rows
CTU_DIRS.update({f"ctuk_lofo_{s}": f"data/ctuk_lofo_{s}" for s in ("cchb", "ddos", "okiru")})
# cic  - an earlier CIC split used by the lambda ablation (vocab.json and results only)
# cicr - main CIC split, 04_make_splits_ciciot23.py --name cicr --order-by-rowid
CIC_DIRS = {"cic": "data/cic", "cicr": "data/cicr", "cicr_alt": "data/cicr_alt"}
# ton - TON_IoT network subset (07_make_splits_ton.py); z-scored like CIC, so it shares the CIC
# loader branch with its own concept rules
TON_DIRS = {"ton": "data/ton"}
# tond_* - TON_IoT device-disjoint splits (07_make_splits_ton.py --holdout-attacker/--holdout-normal)
TON_DIRS.update({n: f"data/{n}" for n in ("tond_a30", "tond_a31")})
CIC_DIRS.update(TON_DIRS)
CIC_DIRS.update({f"cicr_lofo_{s}": f"data/cicr_lofo_{s}" for s in ("ddossyn", "mirai", "recon", "vuln")})
DATASETS = [*CTU_DIRS, *CIC_DIRS]


def is_ton(dataset: str) -> bool:
    return dataset in TON_DIRS


def is_ctu(dataset: str) -> bool:
    return dataset in CTU_DIRS


def is_cic(dataset: str) -> bool:
    return dataset in CIC_DIRS


def data_dir(dataset: str) -> Path:
    if dataset in CTU_DIRS:
        return ROOT / CTU_DIRS[dataset]
    if dataset in CIC_DIRS:
        return ROOT / CIC_DIRS[dataset]
    raise ValueError(f"Unknown dataset: {dataset}")


def load_vocab(dataset: str) -> dict:
    return json.loads((data_dir(dataset) / "vocab.json").read_text())


def nesy_feature_cols(dataset: str) -> list:
    """NeSy-NIDS input columns: the dataset's list plus any expert-rule features it dropped."""
    vocab = load_vocab(dataset)
    return vocab.get("nesy_feature_cols", vocab["feature_cols"])
