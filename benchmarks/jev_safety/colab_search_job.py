"""Runs on a fresh Colab GPU VM via `colab run` (see search_tc.py, which embeds the inputs as PAYLOAD below).

One VM, many runs: per run, trains a SetFit model on that run's examples with that run's TrainingArguments and returns
P(positive) for every text of the run's `splits` (default: every shared split). Optional per run: `precision: "fp32"`
turns off mixed precision; `frozen: true` skips body training and fits only the logistic-regression head. Prints gzip+base64 JSON between RESULT markers on stdout.
Training runs under bf16 autocast (fp16 AMP on GPUs without bf16); inference runs in fp32.
"""

import base64
import gzip
import json
import os
import subprocess
import sys
import time
import traceback

PAYLOAD = "__PAYLOAD__"
SETFIT = "git+https://github.com/danielkorat/setfit@__SETFIT_REF__"


def main():
    t_start = time.perf_counter()
    if not os.environ.get("JOB_SKIP_INSTALL"):  # set only for a local CPU dry run
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", SETFIT], check=True)
    import numpy as np
    import torch
    from datasets import Dataset

    import setfit
    from setfit import SetFitModel, Trainer, TrainingArguments

    job = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    env = dict(gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu", precision=precision, torch=torch.__version__,
               setfit=setfit.__version__, python=sys.version.split()[0], install_seconds=time.perf_counter() - t_start)
    print(json.dumps(env), file=sys.stderr, flush=True)
    out = dict(env=env, runs={})
    for run in job["runs"]:
        try:
            train = Dataset.from_dict({"text": [x["text"] for x in run["train"]], "label": [int(x["label"]) for x in run["train"]]})
            model = SetFitModel.from_pretrained(run["model"], device=device)
            if run["max_length"]:  # None = keep the model's own max_seq_length
                model.model_body.max_seq_length = run["max_length"]
            prec = run.get("precision", precision)
            args = TrainingArguments(**run["args"], max_length=run["max_length"], seed=run["seed"], report_to="none",
                                     show_progress_bar=False, logging_steps=10_000, save_strategy="no",
                                     use_amp=prec == "fp16")
            trainer = Trainer(model=model, args=args, train_dataset=train)
            sync()
            t0 = time.perf_counter()
            if run.get("frozen"):  # head only, on the pretrained body's embeddings
                model.model_head.fit(model.encode(train["text"], show_progress_bar=False), train["label"])
            else:
                with torch.autocast(device, dtype=torch.bfloat16, enabled=prec == "bf16"):
                    trainer.train()
            sync()
            res = dict(train_seconds=time.perf_counter() - t0, p_pos={})
            classes = [int(c) for c in model.model_head.classes_]
            for split in run.get("splits", list(job["texts"])):
                texts = job["texts"][split]
                proba = np.asarray(model.predict_proba(texts, batch_size=64, show_progress_bar=False))
                if not np.isfinite(proba).all():
                    raise RuntimeError(f"{split}: non-finite probabilities")
                res["p_pos"][split] = [round(float(p), 5) for p in proba[:, classes.index(1)]]
            out["runs"][run["id"]] = res
            print(json.dumps({"run": run["id"], "train_s": round(res["train_seconds"], 1)}), file=sys.stderr, flush=True)
            del model, trainer
        except Exception:  # one bad run (e.g. a model that fails to load) must not lose the others
            out["runs"][run["id"]] = dict(error=traceback.format_exc()[-1500:])
            print(json.dumps({"run": run["id"], "error": traceback.format_exc()[-300:]}), file=sys.stderr, flush=True)
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
