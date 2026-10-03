"""Runs on a fresh Colab GPU VM via `colab run` (see run_colab.py, which embeds the inputs as PAYLOAD below).

Installs SetFit, then per task: trains on the 8+8 Opus-labeled examples, predicts P(positive) for every test case,
and times batch-1 GPU inference. Prints the result as gzip+base64 JSON between RESULT markers on stdout.
Training runs under bf16 autocast (fp16 AMP on GPUs without bf16) unless config precision is fp32; inference runs in fp32.
"""

import base64
import gzip
import json
import os
import statistics
import subprocess
import sys
import time

PAYLOAD = "__PAYLOAD__"
SETFIT = "git+https://github.com/danielkorat/setfit@__SETFIT_REF__"


def bench_latency(fn, texts, warmup=5):
    # same procedure as benchmarks/jev_eval/run_setfit.py: sequential single-item calls
    for t in texts[:warmup]:
        fn([t])
    lat = []
    for t in texts:
        s = time.perf_counter()
        fn([t])
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    return dict(p50_ms=statistics.median(lat), p95_ms=lat[int(0.95 * (len(lat) - 1))], n=len(lat))


def main():
    t_start = time.perf_counter()
    if not os.environ.get("JOB_SKIP_INSTALL"):  # set only for a local CPU dry run
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", SETFIT], check=True)
    import numpy as np
    import torch
    from datasets import Dataset

    from setfit import SetFitModel, Trainer, TrainingArguments

    job = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
    cfg = job["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if cfg.get("precision") == "fp32":  # bf16 autocast hurt mpnet as training got longer (README -> mpnet drop)
        precision = "fp32"
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    import setfit
    env = dict(gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu", precision=precision, torch=torch.__version__,
               setfit=setfit.__version__, python=sys.version.split()[0], install_seconds=time.perf_counter() - t_start)
    print(json.dumps(env), file=sys.stderr, flush=True)
    out = dict(env=env, config=cfg, tasks={})
    for task in job["tasks"]:
        train = Dataset.from_dict({"text": [x["text"] for x in task["train"]],
                                   "label": [int(x["label"]) for x in task["train"]]})
        model = SetFitModel.from_pretrained(cfg["model"], device=device)
        model.model_body.max_seq_length = cfg["max_length"]
        args = TrainingArguments(batch_size=(cfg["batch_size"], 16), num_epochs=(1, 16),
                                 num_iterations=cfg["num_iterations"], max_length=cfg["max_length"], seed=cfg["seed"],
                                 report_to="none", show_progress_bar=False, logging_steps=10_000, save_strategy="no",
                                 use_amp=precision == "fp16")
        trainer = Trainer(model=model, args=args, train_dataset=train)
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16"):
            trainer.train()
        sync()
        train_s = time.perf_counter() - t0
        texts = task["test_texts"]
        t0 = time.perf_counter()
        proba = np.asarray(model.predict_proba(texts, batch_size=64, show_progress_bar=False))
        infer_s = time.perf_counter() - t0
        if not np.isfinite(proba).all():  # a CPU dry run once produced NaN embeddings; never score those
            raise RuntimeError(f"{task['id']}: {int((~np.isfinite(proba)).any(1).sum())} non-finite probabilities")
        classes = [int(c) for c in model.model_head.classes_]
        p_pos = proba[:, classes.index(1)]
        lat = bench_latency(lambda t: model.predict_proba(t, show_progress_bar=False), texts[: cfg["n_latency"]])
        out["n_params_body"] = sum(p.numel() for p in model.model_body.parameters())
        out["tasks"][task["id"]] = dict(train_seconds=train_s, batched_infer_seconds=infer_s, latency=lat,
                                        p_pos=[round(float(p), 5) for p in p_pos])
        print(json.dumps({"task": task["id"], "train_s": round(train_s, 1), "latency": lat}), file=sys.stderr, flush=True)
        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()
    out["wall_seconds"] = time.perf_counter() - t_start
    blob = base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
    try:  # also on the VM's disk, so run_colab.py can fetch it after a lost connection (`colab download`)
        with open("/content/RESULT.txt", "w") as fh:
            fh.write(blob)
    except OSError:  # local dry run: no /content
        pass
    print("RESULT_BEGIN\n" + blob + "\nRESULT_END", flush=True)


if __name__ == "__main__":
    main()
