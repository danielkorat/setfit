"""Download the unlabeled pool files into data/pools/ (wildguardmix is gated: needs `hf auth login` + accepted terms)."""

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

POOLS = Path(__file__).resolve().parent / "data" / "pools"
FILES = [("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "train.json"),
         ("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_train.csv"),
         ("allenai/wildguardmix", "train/wildguard_train.parquet")]

if __name__ == "__main__":
    POOLS.mkdir(parents=True, exist_ok=True)
    for repo, fn in FILES:
        src = hf_hub_download(repo, fn, repo_type="dataset")
        dst = POOLS / f"{repo.split('/')[0]}__{os.path.basename(fn)}"
        shutil.copyfile(os.path.realpath(src), dst)
        print(dst, dst.stat().st_size)
