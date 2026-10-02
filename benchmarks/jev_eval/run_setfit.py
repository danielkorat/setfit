"""SetFit vs Jev / gpt-5.4-mini / gpt-5.6-luna on the exact jev-eval test items (onlyoneaman/jev-eval).

Test items: the 300 committed cases per dataset (optionally the first --n_test of them).
Train items: drawn from each dataset's *train* split, never the split the test items came from,
with exact-text overlaps with the test items removed.

Device: --device auto picks CUDA, then Apple MPS, then CPU. --precision auto picks bf16 on GPUs/CPUs with native
bf16 (fp16 on CUDA GPUs without it) and fp32 elsewhere, including MPS until bf16 is benchmarked there.

CPU acceleration used:
  * training: all cores via OMP/torch threads, tcmalloc on Linux (LD_PRELOAD in run.sh), capped sequence length,
    bf16 autocast only when the CPU has native bf16 (avx512_bf16 / AMX)
  * inference: PyTorch vs OpenVINO and OpenVINO int8, measured at batch size 1 (how Jev and the LLMs were called),
    in a child process without tcmalloc, which crashes OpenVINO's exporter ("double free or corruption").
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset

from setfit import SetFitModel, Trainer, TrainingArguments, get_templated_dataset, sample_dataset

HERE = Path(__file__).resolve().parent
JEV_EVAL = Path(os.environ.get("JEV_EVAL", HERE / "jev-eval"))  # see fetch_jev_eval.sh
BASELINES = ["jev-latest", "gpt-5.4-mini", "gpt-5.6-luna"]


def cpu_flags() -> set[str]:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


FLAGS = cpu_flags()
NATIVE_BF16 = bool({"avx512_bf16", "amx_bf16"} & FLAGS)
RSS_UNIT = 2**30 if sys.platform == "darwin" else 2**20  # ru_maxrss is bytes on macOS, KiB on Linux


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_precision(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device == "cpu":
        return "bf16" if NATIVE_BF16 else "fp32"
    return "fp32"


def enron_text(r: dict) -> str:
    # identical to jev_eval/datasets.py
    return f"Subject: {r.get('subject') or ''}\n\n{(r.get('message') or r.get('text') or '')[:4000]}"


SPECS = {
    "enron_spam": dict(hf="SetFit/enron_spam", kw={}, text=enron_text, label=lambda r, names: r["label_text"],
                       template="This email is {}"),
    "sst2": dict(hf="stanfordnlp/sst2", kw={}, text=lambda r: r["sentence"], label=lambda r, names: names[r["label"]],
                 template="This movie review is {}"),
    "ag_news": dict(hf="fancyzhx/ag_news", kw={}, text=lambda r: r["text"], label=lambda r, names: names[r["label"]],
                    template="This news article is about {}"),
    "banking77": dict(hf="legacy-datasets/banking77", kw={"revision": "refs/convert/parquet"}, text=lambda r: r["text"],
                      label=lambda r, names: names[r["label"]], template="The customer wants to {}"),
}


def norm_label(s: str) -> str:
    # jev-eval renders banking77 labels as "activate my card"; HF uses "activate_my_card"
    return s.replace("_", " ").strip()


def load_cases(name: str, n_test: int | None):
    cases = [json.loads(l) for l in open(JEV_EVAL / "cases" / f"{name}.jsonl")]
    if n_test:
        cases = cases[:n_test]
    q = cases[0]["state"]["questions"]["label"]
    labels = list(q["labels"].keys())
    descriptions = q["labels"]
    return cases, labels, descriptions


def load_baselines(name: str, ids: set[str]) -> dict:
    out = {}
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(JEV_EVAL / "cases" / f"{name}.jsonl")}
    for b in BASELINES:
        p = JEV_EVAL / "results" / f"{name}.{b}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r["id"] in ids]
        correct, lat, conf_ok, conf_n = [], [], 0, 0
        for r in rows:
            a = r["answers"]["label"]
            ok = norm_label(str(a.get("choice"))) == norm_label(str(cases[r["id"]]["expected"]["label"]))
            correct.append(ok)
            m = r["answers"].get("_meta") or a.get("_meta") or {}
            if "latency_ms" in m:
                lat.append(m["latency_ms"])
            if a.get("confidence") is not None and a["confidence"] >= 0.9:
                conf_n += 1
                conf_ok += ok
        out[b] = dict(n=len(rows), accuracy=float(np.mean(correct)) if correct else None,
                      p50_latency_ms=statistics.median(lat) if lat else None,
                      confident_share=conf_n / len(rows) if rows else None,
                      confident_accuracy=conf_ok / conf_n if conf_n else None)
    return out


def load_train(name: str, labels: list[str], test_texts: set[str]) -> Dataset:
    spec = SPECS[name]
    ds = load_dataset(spec["hf"], split="train", **spec["kw"])
    feat = ds.features.get("label")
    names = [norm_label(n) for n in feat.names] if hasattr(feat, "names") else None
    lab2id = {norm_label(l): i for i, l in enumerate(labels)}
    texts, ys = [], []
    for r in ds:
        t = spec["text"](r)
        if t in test_texts or not t.strip():
            continue
        y = norm_label(str(spec["label"](r, names)))
        if y not in lab2id:
            raise ValueError(f"{name}: train label {y!r} not in test label set")
        texts.append(t)
        ys.append(lab2id[y])
    return Dataset.from_dict({"text": texts, "label": ys})


def zero_shot_train(name: str, labels: list[str], descriptions: dict, per_class: int) -> Dataset:
    # SetFit's zero-shot recipe: templated synthetic sentences from the label names. We use the same one-line
    # label descriptions Jev and the LLMs were given, so every system receives the same label information.
    cands = []
    for l in labels:
        d = descriptions[l]
        cands.append(l if norm_label(d).lower() == norm_label(l).lower() else f"{l} ({d})")
    ds = get_templated_dataset(candidate_labels=cands, sample_size=per_class, template=SPECS[name]["template"])
    # get_templated_dataset labels candidates by index, matching `labels` order
    return ds


def bench_latency(fn, texts: list[str], warmup: int = 5) -> dict:
    for t in texts[:warmup]:
        fn([t])
    lat = []
    for t in texts:
        s = time.perf_counter()
        fn([t])
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    return dict(p50_ms=statistics.median(lat), p95_ms=lat[int(0.95 * (len(lat) - 1))], n=len(lat))


def openvino_predictors(model: SetFitModel, workdir: Path, max_length: int, int8: bool):
    """OpenVINO body (optionally int8 weight-compressed for VNNI) + the trained sklearn head."""
    from optimum.intel import OVModelForFeatureExtraction, OVWeightQuantizationConfig

    body_dir = workdir / "body"
    model.model_body.save(str(body_dir))
    kw = {}
    if int8:
        kw["quantization_config"] = OVWeightQuantizationConfig(bits=8)
    ov = OVModelForFeatureExtraction.from_pretrained(str(body_dir), export=True, **kw)
    tok = model.model_body.tokenizer
    pool = model.model_body[1]
    pooling = pool.pooling_mode if hasattr(pool, "pooling_mode") else pool.get_pooling_mode_str()
    if isinstance(pooling, (list, tuple)):
        pooling = pooling[0]
    normalize = any(type(m).__name__ == "Normalize" for m in model.model_body)

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out = ov(**enc).last_hidden_state
        if pooling == "cls":
            e = out[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            e = (out * m).sum(1) / m.sum(1)
        if normalize:
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        return e.numpy()

    head = model.model_head
    return lambda texts: head.predict_proba(embed(texts))


def run_one(name: str, shots: int, seed: int, args) -> dict:
    cases, labels, descriptions = load_cases(name, args.n_test)
    test_texts = [c["state"]["text"] for c in cases]
    y_true = np.array([labels.index(c["expected"]["label"]) for c in cases])

    if shots == 0:
        train = zero_shot_train(name, labels, descriptions, args.zs_per_class)
    else:
        full = load_train(name, labels, set(test_texts))
        train = sample_dataset(full, label_column="label", num_samples=shots, seed=seed)

    device = pick_device(args.device)
    precision = pick_precision(args.precision, device)
    model = SetFitModel.from_pretrained(args.model, labels=labels, device=device)
    model.model_body.max_seq_length = args.max_length
    targs = TrainingArguments(
        batch_size=(args.batch_size, 16),
        num_epochs=(1, 16),
        num_iterations=args.num_iterations,
        # a positive max_steps makes the trainer loop the data up to it, so only ever use it to cut a run short
        max_steps=min(args.max_steps, -(-len(train) * args.num_iterations * 2 // args.batch_size))
        if args.max_steps > 0 else -1,
        max_length=args.max_length,
        seed=seed,
        report_to="none",
        show_progress_bar=False,
        logging_steps=10_000,
        save_strategy="no",
        use_amp=precision == "fp16",  # the trainer's AMP path (with loss scaling) is fp16
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train)
    t0 = time.perf_counter()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
        trainer.train()
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = np.asarray(model.predict_proba(test_texts, batch_size=64, show_progress_bar=False))
    infer_s = time.perf_counter() - t0
    pred = proba.argmax(1)
    conf = proba.max(1)
    acc = float((pred == y_true).mean())
    hi = conf >= 0.9

    res = dict(dataset=name, shots=shots, seed=seed, model=args.model, device=device, precision=precision, n_train=len(train), n_test=len(cases),
               accuracy=acc, confident_share=float(hi.mean()),
               confident_accuracy=float((pred[hi] == y_true[hi]).mean()) if hi.any() else None,
               train_seconds=train_s, batched_infer_ms_per_item=infer_s / len(cases) * 1000,
               predictions=[dict(id=c["id"], pred=labels[p], proba=float(q)) for c, p, q in zip(cases, pred, conf)])

    if args.latency and shots >= args.latency_min_shots:
        # OpenVINO crashes under tcmalloc ("double free or corruption"), so every inference backend is timed
        # in a child process without LD_PRELOAD; the trained model is handed over through disk.
        model_dir = Path(args.out) / f"model_{name}_{shots}_{seed}"
        model.save_pretrained(str(model_dir))
        job = dict(model_dir=str(model_dir), texts=test_texts, y_true=y_true.tolist(), n_latency=args.latency,
                   max_length=args.max_length, device=device)
        job_path = model_dir / "latency_job.json"
        job_path.write_text(json.dumps(job))
        env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        proc = subprocess.run([sys.executable, __file__, "--latency_job", str(job_path)], env=env,
                              capture_output=True, text=True)
        out_path = model_dir / "latency_result.json"
        if out_path.exists():
            res.update(json.loads(out_path.read_text()))
        else:
            res["latency_error"] = (proc.stderr or "")[-800:]
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT
    res["peak_rss_latency_child_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / RSS_UNIT
    return res


def latency_job(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    model = SetFitModel.from_pretrained(job["model_dir"], device=job.get("device", "cpu"))
    model.model_body.max_seq_length = job["max_length"]
    texts, y_true = job["texts"], np.array(job["y_true"])
    lat_texts = texts[: job["n_latency"]]
    out = {f"latency_torch_{model.device.type}_fp32": bench_latency(lambda t: model.predict_proba(t, show_progress_bar=False), lat_texts)}
    for tag, int8 in (("openvino_fp32", False), ("openvino_int8", True)):
        try:
            f = openvino_predictors(model, Path(job["model_dir"]) / f"ov_{tag}", job["max_length"], int8)
            out[f"latency_{tag}"] = bench_latency(f, lat_texts)
            out[f"accuracy_{tag}"] = float((np.asarray(f(texts)).argmax(1) == y_true).mean())
        except Exception as e:  # keep the run even if an exporter fails
            out[f"latency_{tag}"] = dict(error=repr(e)[:300])
    (Path(job["model_dir"]) / "latency_result.json").write_text(json.dumps(out))


    return res


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--latency_job":
        torch.set_num_threads(os.cpu_count())
        return latency_job(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="enron_spam,sst2,ag_news,banking77")
    ap.add_argument("--shots", default="0,8")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n_test", type=int, default=0, help="first N of the 300 committed cases (0 = all)")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_iterations", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=-1, help="cap on body steps (only ever shortens a run)")
    ap.add_argument("--zs_per_class", type=int, default=8)
    ap.add_argument("--latency", type=int, default=50, help="items for batch-1 latency (0 = skip)")
    ap.add_argument("--latency_min_shots", type=int, default=1, help="only measure latency for runs with >= this many shots")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    env = dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
               cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               mps=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
               native_bf16=NATIVE_BF16, threads=torch.get_num_threads(), torch=torch.__version__,
               ld_preload=os.environ.get("LD_PRELOAD", ""), args=vars(args))
    print(json.dumps(env), flush=True)
    for name in args.datasets.split(","):
        ids = {c["id"] for c in load_cases(name, args.n_test)[0]}
        base = load_baselines(name, ids)
        for shots in map(int, args.shots.split(",")):
            for seed in map(int, args.seeds.split(",")):
                r = run_one(name, shots, seed, args)
                r["baselines"] = base
                r["env"] = env
                fn = Path(args.out) / f"{name}.shots{shots}.seed{seed}.json"
                fn.write_text(json.dumps(r, indent=1))
                brief = {k: v for k, v in r.items() if k not in ("predictions", "env", "baselines")}
                print(json.dumps(brief), "| baselines:", {k: round(v["accuracy"], 3) for k, v in base.items()},
                      flush=True)


if __name__ == "__main__":
    main()
