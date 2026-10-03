"""Ship one seed's training sets + all test inputs to a Colab GPU, train/predict there, score locally with jsb.

    upstream/.venv/bin/python run_colab.py --seed 0 [--gpus L4,T4,A100]
    upstream/.venv/bin/python run_colab.py --seed 0 --labels gold --model BAAI/bge-base-en-v1.5 --shots 32

--labels opus (default): inputs are work/selections/seed<s>/<task>.json (label_opus.py, 8 per class); outputs
(committed, no dataset text) are results/seed<s>.json (metrics per task via jsb.metrics.stats) and
results/predictions/seed<s>.json (P(positive) per case id).
--labels gold: no claude -p calls; --shots positives and negatives per task are taken from the same pools by their
gold label (pool shuffled with Random(seed), first k of each class, so smaller k is a subset of larger k). Outputs go
to results/gold/<model name>_k<shots>/ with the same layout. These runs use human labels: keep them apart from the
"0 human labels" column.
HarmBench response: the selected training items are removed from that seed's test set before scoring.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

from tasks import HERE, TASKS, Item, Task, input_text, load_pool, load_test

from jsb.metrics.stats import confusion  # noqa: E402  (tasks.py puts upstream on sys.path)
from jsb.workflow.decide import BOOLEAN_TRUE_THRESHOLD  # noqa: E402  (0.5, upstream's threshold)

CONFIG = dict(model="sentence-transformers/all-mpnet-base-v2", per_class=8, batch_size=16, max_length=384,
              num_iterations=20, n_latency=100, labels="opus", precision="auto")
SETFIT_REF = "a000da034c7d15328a119424dfcb769231d945d2"  # danielkorat/setfit main when this benchmark was added; no library changes since
WORK = HERE / "work"
RESULTS = HERE / "results"


def gold_selection(task: Task, seed: int, k: int) -> dict:
    """First k gold-positive and k gold-negative pool items in Random(seed) order (no Opus labels involved)."""
    pool = [it for it in load_pool(task, load_test(task)) if it.gold is not None]
    random.Random(seed).shuffle(pool)
    pos = [it for it in pool if it.gold][:k]
    neg = [it for it in pool if not it.gold][:k]
    assert len(pos) == len(neg) == k, f"{task.id}: pool has {len(pos)} positives, {len(neg)} negatives, need {k}"
    last = max(pool.index(pos[-1]), pool.index(neg[-1]))
    return dict(n_labeled=last + 1, n_refused=0,
                train=[dict(id=it.id, prompt=it.prompt, response=it.response, label=it.gold) for it in pos + neg])


def out_dir(cfg: dict) -> tuple[Path, str]:
    """(results dir, work-file prefix) for a config: the Opus run keeps its original paths."""
    if cfg["labels"] == "opus":
        return RESULTS, ""
    tag = f"{cfg['model'].split('/')[-1]}_k{cfg['per_class']}" + ("_fp32" if cfg["precision"] == "fp32" else "")
    return RESULTS / "gold" / tag, f"gold_{tag}_"


def build_job(seed: int, cfg: dict) -> tuple[dict, dict]:
    tasks, meta = [], {}
    for t in TASKS:
        if cfg["labels"] == "gold":
            sel = gold_selection(t, seed, cfg["per_class"])
        else:
            sel = json.loads((WORK / "selections" / f"seed{seed}" / f"{t.id}.json").read_text())
        train = [Item(x["id"], x["prompt"], x["response"], x["label"]) for x in sel["train"]]
        train_ids = {x.id for x in train}
        assert len(train_ids) == 2 * cfg["per_class"] and sum(x.gold for x in train) == cfg["per_class"]
        test = [it for it in load_test(t) if it.id not in train_ids]  # only HarmBench pools overlap the test set
        assert len(test) == t.published_n - (len(train_ids) if t.id == "hb_harmful_response" else 0)
        tasks.append(dict(id=t.id, train=[dict(text=input_text(t, x), label=x.gold) for x in train],
                          test_texts=[input_text(t, x) for x in test]))
        meta[t.id] = dict(test=test, train_ids=[x.id for x in train], train_labels=[x.gold for x in train],
                          n_labeled=sel["n_labeled"], n_refused=sel.get("n_refused", 0))
    return dict(config=cfg | dict(seed=seed), tasks=tasks), meta


def run_on_colab(job: dict, seed: int, gpus: list[str], prefix: str, template: str = "colab_job.py",
                 cap_seconds: int = 5400 + 600) -> dict:
    payload = base64.b64encode(gzip.compress(json.dumps(job).encode())).decode()
    src = (HERE / template).read_text().replace("__PAYLOAD__", payload).replace("__SETFIT_REF__", SETFIT_REF)
    script = WORK / f"{prefix}colab_seed{seed}.py"
    script.write_text(src)
    print(f"payload {len(payload) / 1e6:.1f} MB", flush=True)
    for gpu in gpus:
        print(f"colab run --gpu {gpu} ...", flush=True)
        try:  # the CLI once hung after creating a session and never executed the script (3 Oct 2026): cap it
            p = subprocess.run(["colab", "run", "--gpu", gpu, "--timeout", "5400", str(script)],
                               capture_output=True, text=True, timeout=cap_seconds)
        except subprocess.TimeoutExpired as e:
            print(f"{gpu}: colab run still running after {e.timeout:.0f} s, killed", flush=True)
            subprocess.run(["colab", "sessions"])  # prunes the stale local session and lists what the server still holds
            continue
        (WORK / f"{prefix}colab_seed{seed}_{gpu}.log").write_text(p.stderr + "\n--- stdout ---\n" + p.stdout[:5000])
        if "RESULT_BEGIN" in p.stdout:
            blob = p.stdout.split("RESULT_BEGIN\n", 1)[1].split("\nRESULT_END", 1)[0]
            return json.loads(gzip.decompress(base64.b64decode(blob)))
        print(f"{gpu} failed (exit {p.returncode}):\n{p.stderr[-1500:]}", flush=True)
        if "Traceback" in p.stdout + p.stderr:  # a script error, not a GPU-allocation failure: don't retry elsewhere
            break
    sys.exit(f"seed {seed}: no result")


def score(seed: int, out: dict, meta: dict, results: Path) -> dict:
    res = dict(seed=seed, config=out["config"], env=out["env"], wall_seconds=out["wall_seconds"],
               n_params_body=out["n_params_body"], tasks={})
    preds = {}
    for t in TASKS:
        r, m = out["tasks"][t.id], meta[t.id]
        assert len(r["p_pos"]) == len(m["test"])
        pairs = [(bool(it.gold), p >= BOOLEAN_TRUE_THRESHOLD) for it, p in zip(m["test"], r["p_pos"])]
        c = confusion(pairs)
        res["tasks"][t.id] = dict(n=c.n, tp=c.tp, fp=c.fp, tn=c.tn, fn=c.fn,
                                  f1=c.f1.value, f1_ci=list(c.f1.ci), precision=c.precision.value, recall=c.recall.value,
                                  train_seconds=r["train_seconds"], batched_infer_seconds=r["batched_infer_seconds"],
                                  latency=r["latency"], n_labeled=m["n_labeled"], n_refused=m["n_refused"],
                                  train_ids=m["train_ids"], train_labels=m["train_labels"])
        preds[t.id] = {it.id: p for it, p in zip(m["test"], r["p_pos"])}
        print(f"{t.id}: n={c.n} F1={c.f1.value:.4f} CI=({c.f1.ci[0]:.4f}, {c.f1.ci[1]:.4f}) "
              f"train={r['train_seconds']:.1f}s p50={r['latency']['p50_ms']:.1f}ms", flush=True)
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    (results / f"seed{seed}.json").write_text(json.dumps(res, indent=1))
    (results / "predictions" / f"seed{seed}.json").write_text(json.dumps(preds))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpus", default="L4,T4,A100")
    ap.add_argument("--labels", choices=["opus", "gold"], default="opus")
    ap.add_argument("--model", default=CONFIG["model"])
    ap.add_argument("--shots", type=int, default=CONFIG["per_class"], help="training examples per class")
    ap.add_argument("--precision", choices=["auto", "fp32"], default="auto",
                    help="auto: bf16 training where supported (fp16 otherwise); fp32: no mixed precision (bf16 hurt mpnet, see README)")
    ap.add_argument("--rescore", action="store_true", help="re-score work/<prefix>colab_out_seed<s>.json without Colab")
    ap.add_argument("--dry", action="store_true", help="build the job and print its sizes; no Colab")
    a = ap.parse_args()
    if a.labels == "opus" and a.shots != CONFIG["per_class"]:
        ap.error(f"--labels opus has {CONFIG['per_class']} Opus-labeled examples per class; other --shots need --labels gold")
    cfg = CONFIG | dict(model=a.model, per_class=a.shots, labels=a.labels, precision=a.precision)
    results, prefix = out_dir(cfg)
    job, meta = build_job(a.seed, cfg)
    if a.dry:
        for t in job["tasks"]:
            print(f"{t['id']}: train {len(t['train'])} (pos {sum(x['label'] for x in t['train'])}) "
                  f"test {len(t['test_texts'])} scanned {meta[t['id']]['n_labeled']}")
        print(f"-> {results.relative_to(HERE)}/seed{a.seed}.json")
        return
    raw = WORK / f"{prefix}colab_out_seed{a.seed}.json"
    if not a.rescore:
        raw.write_text(json.dumps(run_on_colab(job, a.seed, a.gpus.split(","), prefix)))
    out = json.loads(raw.read_text())
    print(json.dumps(out["env"]))
    score(a.seed, out, meta, results)


if __name__ == "__main__":
    main()
