"""Why does all-mpnet-base-v2 get worse with more gold examples (gold pilot: 8 -> 32 per class)? One Colab job.

    upstream/.venv/bin/python diag_mpnet.py [--dry] [--rescore]

Same gold examples, hyperparameters and test sets as the pilot (run_colab.py --labels gold, seed 0), on the two
prompt tasks where mpnet dropped; each variant changes one thing against mpnet at 32 per class:
fp32 instead of bf16, 5 pair iterations (the same 640 pairs as 8 per class at 20), or a frozen body (head only).
bge-base-en-v1.5 at 32 per class is the control (it improved with more examples). Scored on the test sets.

Output: results/diag/mpnet_drop.json (metrics only).
"""

from __future__ import annotations

import argparse
import json

from run_colab import WORK, gold_selection, run_on_colab
from search_tc import FS_ARGS, auroc, metrics
from tasks import BY_ID, HERE, Item, input_text, load_test

from jsb.metrics.stats import confusion  # noqa: E402  (tasks.py puts upstream on sys.path)

TASKS = ["tc_toxic", "wg_prompt_harmful"]
MPNET, BGE = "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"
# variant -> (model, per class, overrides)
VARIANTS = {
    "mpnet_k8": (MPNET, 8, {}),
    "mpnet_k32": (MPNET, 32, {}),
    "mpnet_k32_fp32": (MPNET, 32, dict(precision="fp32")),
    "mpnet_k32_it5": (MPNET, 32, dict(args=FS_ARGS | dict(num_iterations=5))),
    "mpnet_k32_frozen": (MPNET, 32, dict(frozen=True)),
    "mpnet_k8_frozen": (MPNET, 8, dict(frozen=True)),
    "bge_k32_fp32": (BGE, 32, dict(precision="fp32")),
    "bge_k32_frozen": (BGE, 32, dict(frozen=True)),
}
PILOT = {"mpnet_k8": "all-mpnet-base-v2_k8", "mpnet_k32": "all-mpnet-base-v2_k32"}  # determinism check


def build() -> list[dict]:
    runs = []
    for tid in TASKS:
        task = BY_ID[tid]
        for name, (model, k, over) in VARIANTS.items():
            sel = gold_selection(task, 0, k)
            runs.append(dict(id=f"{tid}|{name}", model=model, seed=0, max_length=384, args=FS_ARGS, splits=[tid],
                             train=[dict(text=input_text(task, Item(x["id"], x["prompt"], x["response"], x["label"])),
                                         label=x["label"]) for x in sel["train"]],
                             train_ids=[x["id"] for x in sel["train"]]) | over)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    tests = {tid: load_test(BY_ID[tid]) for tid in TASKS}
    runs = build()
    for r in runs:  # HarmBench is not in TASKS, so no training item can be in a test set
        assert not set(r["train_ids"]) & {it.id for it in tests[r["splits"][0]]}
    if a.dry:
        for r in runs:
            print(r["id"], r["model"].split("/")[-1], len(r["train"]), sum(x["label"] for x in r["train"]),
                  r["args"].get("num_iterations"), r.get("precision", "auto"), r.get("frozen", False))
        return
    raw = WORK / "diag_mpnet_out.json"
    if not a.rescore:
        job = dict(texts={tid: [input_text(BY_ID[tid], it) for it in tests[tid]] for tid in TASKS},
                   runs=[{k: r[k] for k in ("id", "model", "seed", "max_length", "args", "train", "splits", "precision", "frozen")
                          if k in r} for r in runs])
        raw.write_text(json.dumps(run_on_colab(job, 0, ["L4"], "diag_mpnet_", template="colab_search_job.py",
                                               cap_seconds=2400)))
    out = json.loads(raw.read_text())
    print(json.dumps(out["env"]))
    res = {}
    for r in runs:
        tid, name = r["id"].split("|")
        o = out["runs"][r["id"]]
        if "error" in o:
            print(f"{r['id']}: ERROR {o['error'].strip().splitlines()[-1]}")
            res[r["id"]] = dict(error=o["error"])
            continue
        y, p = [bool(it.gold) for it in tests[tid]], o["p_pos"][tid]
        c = confusion([(a_, b >= 0.5) for a_, b in zip(y, p)])
        m = metrics(y, p, 0.5)
        res[r["id"]] = dict(m, auroc=auroc(y, p), f1_ci=list(c.f1.ci), precision=c.precision.value, recall=c.recall.value,
                            train_seconds=o["train_seconds"], train_ids=r["train_ids"])
        pilot = ""
        if name in PILOT:
            ref = json.loads((HERE / "results" / "gold" / PILOT[name] / "seed0.json").read_text())["tasks"][tid]["f1"]
            pilot = f"  (pilot {100 * ref:.1f})"
        print(f"{tid:18s} {name:18s} F1 {100 * m['f1']:5.1f}  P {100 * c.precision.value:5.1f}  R {100 * c.recall.value:5.1f}  "
              f"AUROC {100 * res[r['id']]['auroc']:5.1f}  train {o['train_seconds']:5.1f}s{pilot}")
    (HERE / "results" / "diag").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "diag" / "mpnet_drop.json").write_text(json.dumps(dict(env=out["env"], runs=res), indent=1))


if __name__ == "__main__":
    main()
