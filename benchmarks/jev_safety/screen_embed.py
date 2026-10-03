"""Step 1 of IMPROVEMENT_PLAN.md: frozen-embedding screen of candidate SetFit bodies (one Colab job, colab_embed_job.py).

    upstream/.venv/bin/python screen_embed.py [--dry] [--rescore] [--models embeddinggemma-300m --tag gemma]

Per task and seed 0-4, the gold training sample is run_colab.gold_selection(task, seed, 32): the first 8 / 16 / 32 of
each class are the 8 / 16 / 32-per-class sets (nested). Each model embeds every text once; a logistic-regression head
is fit per (labels per class, seed) on the stored embeddings. HarmBench response trains on test items, so each run's
training items are removed from its test set. Metrics per run: F1 at 0.5, F1 at a threshold chosen from the k labels
alone (out-of-fold), F1 at two known-rate thresholds, best possible F1, AUPRC, AUROC.

Output: results/screen/embed_<tag>.json (default tag step1) (metrics only, no dataset text).
"""

from __future__ import annotations

import argparse
import json
import statistics

from run_colab import WORK, gold_selection, run_on_colab
from tasks import TASKS, HERE, Item, input_text, load_test

from jsb.metrics.stats import confusion  # noqa: E402  (tasks.py puts upstream on sys.path)

SEEDS = [0, 1, 2, 3, 4]
PER_CLASS = [8, 16, 32]
MAX_LENGTH = 512
# task instruction for instruction-following embedders (Qwen3-Embedding's "Instruct: ...\nQuery:" format)
INSTRUCTIONS = {
    "wg_prompt_harmful": "Classify whether the user request is harmful",
    "ag2_unsafe_prompt": "Classify whether the user prompt is unsafe",
    "tc_toxic": "Classify whether the user message to a chatbot is toxic",
    "hb_harmful_response": "Classify whether the assistant response gives harmful help to the request",
    "ag2_unsafe_response": "Classify whether the assistant response is unsafe",
}
MODELS = [
    dict(name="all-mpnet-base-v2", id="sentence-transformers/all-mpnet-base-v2"),
    dict(name="bge-base-en-v1.5", id="BAAI/bge-base-en-v1.5"),
    dict(name="bge-large-en-v1.5", id="BAAI/bge-large-en-v1.5"),
    dict(name="gte-modernbert-base", id="Alibaba-NLP/gte-modernbert-base"),
    dict(name="gte-large-en-v1.5", id="Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True),
    dict(name="Qwen3-Embedding-0.6B", id="Qwen/Qwen3-Embedding-0.6B", instruction=True),
    dict(name="embeddinggemma-300m", id="google/embeddinggemma-300m", prefix="task: classification | query: "),  # gated
    # non-commercial license (CC BY-NC): a reference point only, never the business setting
    dict(name="jina-embeddings-v3", id="jinaai/jina-embeddings-v3", trust_remote_code=True, encode_kwargs=dict(task="classification")),
    # last: a CUDA device-side error here (seen with transformers 5) poisons the GPU for every later model
    dict(name="nomic-embed-text-v1.5", id="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, prefix="classification: "),
]
HF_TOKEN_FILE = WORK / "hf_token"  # gitignored; the operator pastes a read token here for the gated Gemma model
CHECK = dict(model="all-mpnet-base-v2", k=32, seed=0)  # P(positive) returned for this run, re-scored with jsb locally


def build(models: list[dict]) -> tuple[dict, dict]:
    tasks, tests = {}, {}
    for t in TASKS:
        test = load_test(t)
        tests[t.id] = test
        texts = {it.id: input_text(t, it) for it in test}
        train = {}
        for seed in SEEDS:
            sel = gold_selection(t, seed, max(PER_CLASS))["train"]
            for x in sel:
                texts[x["id"]] = input_text(t, Item(x["id"], x["prompt"], x["response"], x["label"]))
            train[str(seed)] = dict(pos=[x["id"] for x in sel if x["label"]], neg=[x["id"] for x in sel if not x["label"]])
        tasks[t.id] = dict(texts=texts, test_ids=[it.id for it in test], test_y=[bool(it.gold) for it in test],
                           rate=sum(it.gold for it in test) / len(test), train=train, instruction=INSTRUCTIONS[t.id])
    return dict(tasks=tasks, models=models, per_class=PER_CLASS, max_length=MAX_LENGTH, check=CHECK), tests


def summarize(out: dict) -> dict:
    """model -> k -> metric -> (mean, std) over seeds of the five-task average."""
    summ = {}
    for name, m in out["models"].items():
        if "error" in m:
            continue
        summ[name] = {}
        for k in PER_CLASS:
            per_seed = {met: [statistics.mean(m["runs"][f"{t.id}|k{k}|s{s}"][met] for t in TASKS) for s in SEEDS]
                        for met in ("f1", "f1_oof", "f1_prior", "f1_topk", "f1_oracle", "auprc", "auroc")}
            summ[name][k] = {met: (100 * statistics.mean(v), 100 * statistics.stdev(v)) for met, v in per_seed.items()}
            summ[name][k]["per_task_f1"] = {t.id: 100 * statistics.mean(m["runs"][f"{t.id}|k{k}|s{s}"]["f1"] for s in SEEDS)
                                            for t in TASKS}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--models", help="comma-separated model names to run (default: all)")
    ap.add_argument("--tag", default="step1", help="output name: results/screen/embed_<tag>.json")
    ap.add_argument("--pip", help="pip install arguments on the VM, e.g. 'sentence-transformers<6 transformers<5 einops'")
    ap.add_argument("--isolate", action="store_true",
                    help="install --pip into a side directory and run each model in its own process (older libraries)")
    a = ap.parse_args()
    models = [m for m in MODELS if not a.models or m["name"] in a.models.split(",")]
    job, tests = build(models)
    if a.pip:
        job["pip"] = a.pip.split()
    if a.isolate:
        assert a.pip, "--isolate needs --pip (the packages to put in the side directory)"
        job.update(isolate=True, self_src=(HERE / "colab_embed_job.py").read_text())
    n_texts = sum(len(t["texts"]) for t in job["tasks"].values())
    print(f"{len(models)} models, {n_texts} texts to embed per model, {len(SEEDS)} seeds x {PER_CLASS} per class")
    if a.dry:
        print("pip:", job.get("pip", "default (latest sentence-transformers)"))
        for tid, t in job["tasks"].items():
            print(tid, len(t["texts"]), "texts, test", len(t["test_ids"]), f"rate {t['rate']:.3f}",
                  "seed0 pos/neg", len(t["train"]["0"]["pos"]), len(t["train"]["0"]["neg"]))
        return
    raw = WORK / f"screen_embed_{a.tag}_out.json"
    if not a.rescore:
        token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else ""
        print("HF token file:", "found, passed to the VM as HF_TOKEN" if token else "empty or missing, gated models will fail")
        raw.write_text(json.dumps(run_on_colab(job, 0, ["L4"], f"screen_embed_{a.tag}_", template="colab_embed_job.py", cap_seconds=4800,
                                               env=dict(HF_TOKEN=token) if token else None)))
    out = json.loads(raw.read_text())
    print(json.dumps(out["env"]))
    for name, m in out["models"].items():
        if "error" in m:
            print(f"{name}: ERROR {m['error'].strip().splitlines()[-1]}")
    # check: the VM's F1 for one run equals upstream's jsb F1 on the returned probabilities
    chk = out["models"].get(CHECK["model"], {}).get("check_p_pos", {})
    for tid, by_case in chk.items():
        gold = {it.id: bool(it.gold) for it in tests[tid]}
        c = confusion([(gold[cid], p >= 0.5) for cid, p in by_case.items()])
        vm = out["models"][CHECK["model"]]["runs"][f"{tid}|k{CHECK['k']}|s{CHECK['seed']}"]["f1"]
        # older runs rounded the check probabilities to 6 decimals, which can move a score onto 0.5
        boundary = sum(abs(p - 0.5) < 1e-6 for p in by_case.values())
        assert abs(c.f1.value - vm) < 1e-9 or (boundary and abs(c.f1.value - vm) < 0.002 * boundary), (tid, c.f1.value, vm)
    print("check: VM F1 == jsb F1 on the check run" if chk else "check: skipped (check model not in this run)")
    summ = summarize(out)
    (HERE / "results" / "screen").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "screen" / f"embed_{a.tag}.json").write_text(json.dumps(dict(
        env=out["env"], wall_seconds=out["wall_seconds"], models=models, seeds=SEEDS, per_class=PER_CLASS,
        max_length=MAX_LENGTH, instructions=INSTRUCTIONS,
        per_model={n: {k: v for k, v in m.items() if k not in ("check_p_pos",)} for n, m in out["models"].items()},
        summary=summ), indent=1))
    print(f"\nfive-task average, mean ± std over {len(SEEDS)} seeds")
    print(f"{'model':24s} k  | F1@0.5      F1 oof      F1 top-k    best F1     AUPRC       AUROC")
    for k in PER_CLASS:
        for name in sorted(summ, key=lambda n: -summ[n][k]["f1"][0]):
            s = summ[name][k]
            print(f"{name:24s} {k:2d} | " + "  ".join(f"{s[met][0]:5.1f}±{s[met][1]:4.1f}" for met in
                                                    ("f1", "f1_oof", "f1_topk", "f1_oracle", "auprc", "auroc")))
        print()


if __name__ == "__main__":
    main()
