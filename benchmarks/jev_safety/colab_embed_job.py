"""Runs on a fresh Colab GPU VM via `colab run` (see screen_embed.py, which embeds the inputs as PAYLOAD below).

Frozen-embedding screen: per model, embeds every test and training text once (fp32, with the model's classification
prefix), then fits a logistic-regression head for each (labels per class, seed) on the stored embeddings and scores it.
Returns metrics only (plus P(positive) for one check run). Prints gzip+base64 JSON between RESULT markers on stdout.
"""

import base64
import gzip
import json
import subprocess
import sys
import time
import traceback

PAYLOAD = "__PAYLOAD__"


def f1_at(y, p, t):
    """Positive-class F1 when predicting positive at p >= t (y, p: numpy arrays)."""
    pred = p >= t
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def scores(y, p, rate):
    """F1 at 0.5, at the known-rate thresholds (logit prior shift; flag the top `rate` share), best possible F1,
    AUPRC and AUROC. `rate` is the test set's positive rate, which a business would have to know or estimate."""
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    y, p = np.asarray(y, bool), np.asarray(p, float)
    pr, rc, _ = precision_recall_curve(y, p)
    top_t = np.sort(p)[::-1][max(round(rate * len(p)), 1) - 1]
    return dict(f1=f1_at(y, p, 0.5), f1_prior=f1_at(y, p, 1 - rate), f1_topk=f1_at(y, p, top_t),
                f1_oracle=float(np.max(np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0))),
                auprc=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)))


def main():
    t_start = time.perf_counter()
    job = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
    # remote-code models (gte-large-en-v1.5, jina-embeddings-v3) break on transformers 5; such jobs pin older versions
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *job.get("pip", ["-U", "sentence-transformers", "einops"])],
                   check=True)
    import numpy as np
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    import transformers

    env = dict(transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0), torch=torch.__version__, sentence_transformers=sentence_transformers.__version__,
               python=sys.version.split()[0], install_seconds=time.perf_counter() - t_start)
    print(json.dumps(env), file=sys.stderr, flush=True)
    out = dict(env=env, models={})
    for m in job["models"]:
        res = out["models"][m["name"]] = dict(runs={})
        try:
            t0 = time.perf_counter()
            model = SentenceTransformer(m["id"], device="cuda", trust_remote_code=m.get("trust_remote_code", False))
            model.max_seq_length = min(job["max_length"], model.max_seq_length or job["max_length"])
            res.update(n_params=sum(p.numel() for p in model.parameters()), max_seq_length=model.max_seq_length,
                       load_seconds=time.perf_counter() - t0)
            for tid, task in job["tasks"].items():
                ids = list(task["texts"])
                if m.get("instruction"):  # Qwen3-Embedding: task instruction in front of every text
                    texts = [f"Instruct: {task['instruction']}\nQuery:{task['texts'][i]}" for i in ids]
                else:
                    texts = [m.get("prefix", "") + task["texts"][i] for i in ids]
                t0 = time.perf_counter()
                emb = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, **m.get("encode_kwargs", {}))
                res.setdefault("embed_seconds", {})[tid] = time.perf_counter() - t0
                if not np.isfinite(emb).all():
                    raise RuntimeError(f"{tid}: non-finite embeddings")
                row = {i: r for i, r in enumerate(ids)}
                pos = {r: i for i, r in row.items()}
                for seed, train in task["train"].items():
                    for k in job["per_class"]:
                        tr = train["pos"][:k] + train["neg"][:k]
                        y_tr = np.asarray([True] * k + [False] * k)
                        tr_set = set(tr)
                        test = [(c, g) for c, g in zip(task["test_ids"], task["test_y"]) if c not in tr_set]
                        X_tr = emb[[pos[c] for c in tr]]
                        head = LogisticRegression(max_iter=1000)  # SetFit's head (sklearn defaults), more iterations
                        head.fit(X_tr, y_tr)
                        p = head.predict_proba(emb[[pos[c] for c, _ in test]])[:, list(head.classes_).index(True)]
                        y = [g for _, g in test]
                        r = scores(y, p, task["rate"])
                        # threshold from the k labels alone: out-of-fold probabilities, F1-best cut (0.5 wins ties)
                        oof = cross_val_predict(LogisticRegression(max_iter=1000), X_tr, y_tr,
                                                cv=StratifiedKFold(4, shuffle=True, random_state=int(seed)), method="predict_proba")[:, 1]
                        t_oof = max(sorted(set(oof.tolist()) | {0.5}), key=lambda t: (f1_at(y_tr, oof, t), -abs(t - 0.5)))
                        r.update(f1_oof=f1_at(np.asarray(y, bool), np.asarray(p), t_oof), t_oof=float(t_oof), n=len(y))
                        res["runs"][f"{tid}|k{k}|s{seed}"] = r
                        if m["name"] == job["check"]["model"] and k == job["check"]["k"] and str(seed) == str(job["check"]["seed"]):
                            res.setdefault("check_p_pos", {})[tid] = {c: float(x) for (c, _), x in zip(test, p)}  # unrounded
                print(json.dumps({"model": m["name"], "task": tid, "embed_s": round(res["embed_seconds"][tid], 1)}), file=sys.stderr, flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:  # a model that fails to load (gated, remote code) must not lose the others
            res["error"] = traceback.format_exc()[-2000:]
            print(json.dumps({"model": m["name"], "error": traceback.format_exc()[-400:]}), file=sys.stderr, flush=True)
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
