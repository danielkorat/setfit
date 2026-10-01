"""Paired benchmark of CosineSimilarityLoss vs SupConLoss vs SINCERELoss on the SetFit paper's test datasets.

Every experiment trains SetFit on the same 10 few-shot splits per dataset (`setfit.utils.load_data_splits`, seeds 0-9)
with the paper's settings (paraphrase-mpnet-base-v2, body LR 2e-5, max 256 tokens, logistic-regression head) and
writes one JSON file per run. Finished runs are skipped, so the script can be stopped and restarted at any time.

Usage:
    python run.py                     # run the whole queue in order
    python run.py --only main-n8      # run one experiment
    python run.py --limit-splits 1 --only main-n8 --datasets emotion   # smoke test
"""

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import sentence_transformers
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import setfit
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit.compat import losses
from setfit.losses import SINCERELoss, SupConLoss
from setfit.utils import TEST_DATASET_TO_METRIC, load_data_splits


MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
LOSSES = {"cosine": losses.CosineSimilarityLoss, "supcon": SupConLoss, "sincere": SINCERELoss}
DATASETS = list(TEST_DATASET_TO_METRIC)  # emotion, SentEval-CR, sst5, ag_news, enron_spam, amazon_counterfactual_en
RUNS_DIR = Path(os.environ.get("SINCERE_RUNS_DIR", Path(__file__).parent / "runs"))
EVAL_BATCH_SIZE = 128
# Paper protocol for CosineSimilarityLoss: 20 iterations of positive + negative pairs per sentence, batch of 16 pairs.
NUM_ITERATIONS = 20
PAIR_BATCH_SIZE = 16


@dataclass
class Experiment:
    name: str
    losses: List[str]
    sample_sizes: List[int]
    # "matched": SupCon/SINCERE get the same number of steps as CosineSimilarityLoss, with up to 32 sentences per step
    # (a cosine step holds 16 pairs = 32 sentences). "setfit_default": SetFit's current behaviour for SupConLoss,
    # one epoch over the few-shot sentences in batches of 16.
    budget: str = "matched"
    temperature: float = 0.07
    precision: str = "bf16"
    splits: List[int] = field(default_factory=lambda: list(range(10)))
    repeat: int = 0  # > 0 reruns an identical configuration to measure run-to-run noise


QUEUE = [
    Experiment("main-n8", ["cosine", "supcon", "sincere"], [8]),
    Experiment("main-n64", ["cosine", "supcon", "sincere"], [64]),
    Experiment("default-budget", ["supcon", "sincere"], [8, 64], budget="setfit_default"),
    *[
        Experiment(f"temp-{t}", ["supcon", "sincere"], [8], temperature=t)
        for t in (0.1, 0.3, 0.5)
    ],
    Experiment("parity-fp32", ["cosine", "supcon", "sincere"], [8], precision="fp32", splits=list(range(5))),
    Experiment("noise-repeat", ["cosine", "supcon", "sincere"], [8], splits=list(range(3)), repeat=1),
]


def enable_bf16_body(body: torch.nn.Module) -> None:
    """Run the transformer body under bf16 autocast and return fp32 embeddings, so every loss is computed in fp32.

    `TrainingArguments(use_amp=True)` does nothing on MPS (Accelerate reports `mixed_precision: no`), hence this.
    """
    forward = body.forward

    def autocast_forward(*args, **kwargs):
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            output = forward(*args, **kwargs)
        output["sentence_embedding"] = output["sentence_embedding"].float()
        return output

    body.forward = autocast_forward


def set_temperature(trainer: Trainer, temperature: float) -> None:
    """Set the temperature of the loss the trainer builds. For SupCon, base_temperature follows it so that its
    (temperature / base_temperature) loss scaling stays 1, exactly as in SINCERE: the denominator is then the only
    difference between the two losses."""
    get_dataset = trainer.get_dataset

    def get_dataset_with_temperature(*args, **kwargs):
        dataset, loss = get_dataset(*args, **kwargs)
        if hasattr(loss, "temperature"):
            loss.temperature = temperature
            if hasattr(loss, "base_temperature"):
                loss.base_temperature = temperature
        return dataset, loss

    trainer.get_dataset = get_dataset_with_temperature


def training_args(loss_name: str, budget: str, num_sentences: int) -> TrainingArguments:
    common = dict(num_epochs=1, seed=42, show_progress_bar=False, report_to="none", logging_strategy="no")
    loss = LOSSES[loss_name]
    if loss_name == "cosine":
        return TrainingArguments(loss=loss, batch_size=PAIR_BATCH_SIZE, num_iterations=NUM_ITERATIONS, **common)
    if budget == "matched":
        cosine_steps = math.ceil(2 * NUM_ITERATIONS * num_sentences / PAIR_BATCH_SIZE)
        return TrainingArguments(loss=loss, batch_size=2 * PAIR_BATCH_SIZE, max_steps=cosine_steps, **common)
    return TrainingArguments(loss=loss, batch_size=PAIR_BATCH_SIZE, **common)


def metrics(y_true, y_pred, primary: str) -> dict:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "matthews_correlation": matthews_corrcoef(y_true, y_pred),
    }
    result["primary"] = result[primary]
    return result


def environment() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=Path(__file__).parent)
    return {
        "git_commit": commit.stdout.strip(),
        "src_dirty": bool(dirty.stdout.strip()),
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "setfit": setfit.__version__,
        "model": MODEL,
    }


def run_one(exp: Experiment, loss_name: str, dataset: str, n: int, split: int, train_data, test_data, env) -> dict:
    model = SetFitModel.from_pretrained(MODEL, device="mps")
    model.model_body.max_seq_length = 256
    if exp.precision == "bf16":
        enable_bf16_body(model.model_body)

    args = training_args(loss_name, exp.budget, len(train_data))
    trainer = Trainer(model=model, args=args, train_dataset=train_data)
    set_temperature(trainer, exp.temperature)

    start = time.perf_counter()
    trainer.train()
    torch.mps.synchronize()
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(test_data["text"], batch_size=EVAL_BATCH_SIZE, as_numpy=True)
    eval_seconds = time.perf_counter() - start

    record = {
        "experiment": asdict(exp),
        "loss": loss_name,
        "dataset": dataset,
        "n": n,
        "split": split,
        "num_train_sentences": len(train_data),
        "num_classes": len(set(train_data["label"])),
        "steps": trainer.st_trainer.state.global_step,
        "train_batch_size": args.embedding_batch_size,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "metric_name": TEST_DATASET_TO_METRIC[dataset],
        "metrics": metrics(np.asarray(test_data["label"]), predictions, TEST_DATASET_TO_METRIC[dataset]),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": env,
    }
    del trainer, model
    gc.collect()
    torch.mps.empty_cache()
    return record


def result_path(exp: Experiment, loss_name: str, dataset: str, n: int, split: int) -> Path:
    suffix = f"_repeat{exp.repeat}" if exp.repeat else ""
    return RUNS_DIR / exp.name / loss_name / dataset / f"n{n}" / f"split{split}{suffix}.json"


def main(only: Optional[List[str]], datasets: List[str], limit_splits: Optional[int]) -> None:
    env = environment()
    if env["src_dirty"]:
        sys.exit("src/ has uncommitted changes: commit them first so every result maps to a git commit.")
    queue = [exp for exp in QUEUE if not only or exp.name in only]
    # For noise-repeat, the first run of each configuration is the matching main-n8 run.
    splits_cache = {}
    for exp in queue:
        splits = exp.splits[:limit_splits] if limit_splits else exp.splits
        for n in exp.sample_sizes:
            # Outer loop over splits, inner over datasets and losses: every completed split yields paired results.
            for split in splits:
                for dataset in datasets:
                    if dataset not in splits_cache:
                        splits_cache[dataset] = load_data_splits(dataset, [8, 64])
                    train_splits, test_data = splits_cache[dataset]
                    train_data = train_splits[f"train-{n}-{split}"]
                    for loss_name in exp.losses:
                        path = result_path(exp, loss_name, dataset, n, split)
                        if path.exists():
                            continue
                        record = run_one(exp, loss_name, dataset, n, split, train_data, test_data, env)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = path.with_suffix(".tmp")
                        tmp.write_text(json.dumps(record, indent=2))
                        os.replace(tmp, path)
                        print(
                            f"[{record['finished_at']}] {exp.name:14} {loss_name:8} {dataset:25} n={n:<3} split={split} "
                            f"{record['metric_name'][:3]}={record['metrics']['primary']:.4f} steps={record['steps']:<5} "
                            f"train={record['train_seconds']:.0f}s eval={record['eval_seconds']:.0f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="experiment names to run (default: whole queue)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--limit-splits", type=int, help="only the first k splits (smoke tests)")
    cli = parser.parse_args()
    main(cli.only, cli.datasets, cli.limit_splits)
