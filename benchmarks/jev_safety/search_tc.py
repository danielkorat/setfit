"""Model / hyperparameter search for ToxicChat prompt (tc_toxic): zero-shot (templated) and gold few-shot SetFit.

    upstream/.venv/bin/python search_tc.py --stage 1 [--dry] [--rescore]

Splits: the ToxicChat train pool (human-annotated, disjoint from the test set) is shuffled with Random(1234); the first
VAL_N items are the validation set and few-shot examples are drawn from the rest (Random(seed), first k of each class).
All runs of a stage share one Colab VM (colab_search_job.py). Every run is scored on validation and test:
toxic-class F1 / macro F1 / accuracy at upstream's 0.5 threshold, AUROC, and toxic F1 at the threshold that maximizes
validation toxic F1. Choosing a setting by validation keeps its test number honest; the test columns are also shown.

Outputs: results/search/tc_stage<n>.json (configs + metrics, no dataset text); raw probabilities in work/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from run_colab import WORK, run_on_colab
from tasks import BY_ID, HERE, input_text, load_pool, load_test

from jsb.metrics.stats import confusion  # noqa: E402  (tasks.py puts upstream on sys.path)

TASK = BY_ID["tc_toxic"]
OUT = HERE / "results" / "search"
VAL_N, VAL_SEED = 1000, 1234
TEMPLATE = "This sentence is {}"  # setfit.get_templated_dataset default
ZS_ARGS = dict(batch_size=32, num_epochs=1)  # docs/source/en/how_to/zero_shot.mdx
FS_ARGS = dict(batch_size=16, num_epochs=1, num_iterations=20)  # the benchmark's run_colab.py CONFIG
MODELS = ["sentence-transformers/paraphrase-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2",
          "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]
# method -> (zero-shot label words (negative, positive) or None, gold examples per class, sample_size per label)
METHODS = {
    "zs_not-toxic": (("not toxic", "toxic"), 0, 8),
    "zs_safe": (("safe", "toxic"), 0, 8),
    "zs_harmless": (("harmless", "harmful"), 0, 8),
    "fs8": (None, 8, 0),
    "fs32": (None, 32, 0),
    "fs8+zs": (("not toxic", "toxic"), 8, 8),
}
STAGES = {1: dict(models=MODELS, methods=list(METHODS), seeds=[0])}


def splits():
    test = load_test(TASK)
    pool = [it for it in load_pool(TASK, test) if it.gold is not None]
    random.Random(VAL_SEED).shuffle(pool)
    return pool[VAL_N:], pool[:VAL_N], test


def templated(words: tuple[str, str], n: int) -> list[dict]:
    """Same examples as setfit.get_templated_dataset(candidate_labels=words, sample_size=n): label 1 = words[1]."""
    return [dict(text=TEMPLATE.format(w), label=y) for y, w in enumerate(words) for _ in range(n)]


def gold_examples(train_pool, seed: int, k: int) -> list[dict]:
    pool = list(train_pool)
    random.Random(seed).shuffle(pool)
    pos = [it for it in pool if it.gold][:k]
    neg = [it for it in pool if not it.gold][:k]
    assert len(pos) == len(neg) == k
    return [dict(id=it.id, text=input_text(TASK, it), label=bool(it.gold)) for it in pos + neg]


def build_runs(stage: dict, train_pool) -> list[dict]:
    runs = []
    for seed in stage["seeds"]:
        for model in stage["models"]:
            for method in stage["methods"]:
                words, k, n_zs = METHODS[method]
                gold = gold_examples(train_pool, seed, k) if k else []
                train = [dict(text=x["text"], label=x["label"]) for x in gold] + (templated(words, n_zs) if words else [])
                runs.append(dict(id=f"{model.split('/')[-1]}|{method}|s{seed}", model=model, method=method, seed=seed,
                                 max_length=384, args=FS_ARGS if k else ZS_ARGS, train=train,
                                 train_ids=[x["id"] for x in gold]))
    return runs


def auroc(y: list[bool], p: list[float]) -> float:
    """Mann-Whitney U / (n_pos * n_neg), ties counted half."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks, i = [0.0] * len(p), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    return (sum(r for r, t in zip(ranks, y) if t) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y: list[bool], p: list[float], t: float) -> dict:
    tp = sum(a and b >= t for a, b in zip(y, p))
    fp = sum(not a and b >= t for a, b in zip(y, p))
    fn = sum(a and b < t for a, b in zip(y, p))
    tn = len(y) - tp - fp - fn
    f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    return dict(f1=f1p, macro_f1=(f1p + f1n) / 2, acc=(tp + tn) / len(y), tp=tp, fp=fp, fn=fn, tn=tn)


def best_threshold(y: list[bool], p: list[float]) -> float:
    """Threshold in {0.5} + validation scores that maximizes toxic F1 (0.5 wins ties)."""
    best = 0.5
    for t in sorted(set(p)):
        if metrics(y, p, t)["f1"] > metrics(y, p, best)["f1"]:
            best = t
    return best


def score(runs: list[dict], out: dict, val, test) -> dict:
    y_val, y_test = [bool(it.gold) for it in val], [bool(it.gold) for it in test]
    res = {}
    for r in runs:
        o = out["runs"][r["id"]]
        row = dict(model=r["model"], method=r["method"], seed=r["seed"], args=r["args"], n_train=len(r["train"]),
                   train_ids=r["train_ids"])
        if "error" in o:
            res[r["id"]] = row | dict(error=o["error"])
            print(f"{r['id']}: ERROR {o['error'].strip().splitlines()[-1]}")
            continue
        pv, pt = o["p_pos"]["val"], o["p_pos"]["test"]
        t = best_threshold(y_val, pv)
        c = confusion([(a, b >= 0.5) for a, b in zip(y_test, pt)])  # upstream's metric, as in the benchmark table
        row |= dict(train_seconds=o["train_seconds"],
                    val=metrics(y_val, pv, 0.5) | dict(auroc=auroc(y_val, pv), tuned_t=t, tuned_f1=metrics(y_val, pv, t)["f1"]),
                    test=metrics(y_test, pt, 0.5) | dict(auroc=auroc(y_test, pt), f1_ci=list(c.f1.ci),
                                                         tuned_f1=metrics(y_test, pt, t)["f1"]))
        assert abs(row["test"]["f1"] - c.f1.value) < 1e-9
        res[r["id"]] = row
    return res


def show(res: dict) -> None:
    print(f"{'run':40s} | val: F1  F1@t  macro  AUROC | test: F1  F1@t  macro  acc  AUROC")
    ok = sorted((v for v in res.values() if "error" not in v), key=lambda v: -max(v["val"]["f1"], v["val"]["tuned_f1"]))
    for v in ok:
        a, b = v["val"], v["test"]
        rid = f"{v['model'].split('/')[-1]}|{v['method']}|s{v['seed']}"
        print(f"{rid:40s} | {100*a['f1']:5.1f} {100*a['tuned_f1']:5.1f} {100*a['macro_f1']:5.1f} {100*a['auroc']:5.1f} | "
              f"{100*b['f1']:5.1f} {100*b['tuned_f1']:5.1f} {100*b['macro_f1']:5.1f} {100*b['acc']:5.1f} {100*b['auroc']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    ap.add_argument("--gpus", default="L4")
    ap.add_argument("--dry", action="store_true", help="build the job and print its sizes; no Colab")
    ap.add_argument("--rescore", action="store_true", help="re-score work/search_tc_stage<n>_out.json without Colab")
    a = ap.parse_args()
    train_pool, val, test = splits()
    assert not {it.id for it in val} & {it.id for it in train_pool}
    runs = build_runs(STAGES[a.stage], train_pool)
    print(f"train pool {len(train_pool)} (pos {sum(it.gold for it in train_pool)}), val {len(val)} "
          f"(pos {sum(it.gold for it in val)}), test {len(test)} (pos {sum(it.gold for it in test)}), runs {len(runs)}")
    if a.dry:
        for r in runs:
            print(r["id"], len(r["train"]), sum(x["label"] for x in r["train"]), r["train"][-1]["text"][:40])
        return
    raw = WORK / f"search_tc_stage{a.stage}_out.json"
    if not a.rescore:
        job = dict(texts=dict(val=[input_text(TASK, it) for it in val], test=[input_text(TASK, it) for it in test]),
                   runs=[{k: r[k] for k in ("id", "model", "seed", "max_length", "args", "train")} for r in runs])
        raw.write_text(json.dumps(run_on_colab(job, 0, a.gpus.split(","), f"search_tc_stage{a.stage}_",
                                               template="colab_search_job.py")))
    out = json.loads(raw.read_text())
    print(json.dumps(out["env"]))
    res = score(runs, out, val, test)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tc_stage{a.stage}.json").write_text(json.dumps(dict(
        env=out["env"], wall_seconds=out["wall_seconds"], val_n=len(val), val_seed=VAL_SEED, template=TEMPLATE,
        val_ids=[it.id for it in val], runs=res), indent=1))
    show(res)


if __name__ == "__main__":
    main()
