"""Turn the per-run JSON records of run.py into RESULTS.md and runs.csv.

Every comparison is paired: two losses are compared on the same (dataset, N, split), so split difficulty cancels out.
Scores are reported x100. Usage: python analyze.py [runs_dir]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).parent
RUNS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs"
DATASETS = ["sst5", "amazon_counterfactual_en", "SentEval-CR", "emotion", "enron_spam", "ag_news"]
# SetFit-MPNet, mean and std over the same 10 splits: Tunstall et al., arXiv 2209.11055v1, Table 2.
PAPER = {
    8: {"sst5": (43.6, 3.0), "amazon_counterfactual_en": (40.3, 11.8), "SentEval-CR": (88.5, 1.9),
        "emotion": (48.8, 4.5), "enron_spam": (90.1, 3.4), "ag_news": (82.9, 2.8)},
    64: {"sst5": (51.9, 0.6), "amazon_counterfactual_en": (61.9, 2.9), "SentEval-CR": (90.4, 0.6),
         "emotion": (76.2, 1.3), "enron_spam": (96.1, 0.8), "ag_news": (88.0, 0.7)},
}
BOOTSTRAP_SAMPLES = 10_000


def load(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.rglob("*.json")):
        r = json.loads(path.read_text())
        exp = r["experiment"]
        rows.append({
            "experiment": exp["name"], "loss": r["loss"], "dataset": r["dataset"], "n": r["n"], "split": r["split"],
            "temperature": exp["temperature"], "precision": exp["precision"], "budget": exp["budget"],
            "repeat": exp["repeat"], "score": 100 * r["metrics"]["primary"], "accuracy": 100 * r["metrics"]["accuracy"],
            "macro_f1": 100 * r["metrics"]["macro_f1"], "steps": r["steps"], "train_seconds": r["train_seconds"],
            "eval_seconds": r["eval_seconds"], "git_commit": r["environment"]["git_commit"],
        })
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Score of loss a minus loss b on each shared (dataset, n, split)."""
    keys = ["dataset", "n", "split"]
    wide = df.pivot_table(index=keys, columns="loss", values="score")
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame(columns=keys + ["delta"])
    wide = wide.dropna(subset=[a, b])
    return (wide[a] - wide[b]).rename("delta").reset_index()


def t_ci(x: np.ndarray) -> tuple:
    if len(x) < 2:
        return (np.nan, np.nan)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - half, x.mean() + half)


def wilcoxon_p(x: np.ndarray) -> float:
    x = x[x != 0]
    return stats.wilcoxon(x).pvalue if len(x) >= 2 else np.nan


def stratified_bootstrap_ci(deltas: pd.DataFrame, seed: int = 0) -> tuple:
    """95% CI of the pooled mean delta, resampling splits within each (dataset, n) cell."""
    rng = np.random.default_rng(seed)
    cells = [g["delta"].to_numpy() for _, g in deltas.groupby(["dataset", "n"])]
    means = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        means[i] = np.mean(np.concatenate([rng.choice(c, size=len(c)) for c in cells]))
    return tuple(np.percentile(means, [2.5, 97.5]))


def holm(pvalues: list) -> list:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return list(adjusted)


def fmt(mean, std=None) -> str:
    if np.isnan(mean):
        return "–"
    return f"{mean:.1f}" if std is None or np.isnan(std) else f"{mean:.1f} ± {std:.1f}"


def comparison_table(df: pd.DataFrame, a: str, b: str) -> tuple:
    """Per-cell paired comparison of loss a vs loss b, plus a pooled line."""
    deltas = paired(df, a, b)
    if deltas.empty:
        return "_no paired runs yet_\n", None
    lines = [f"| N | dataset | Δ mean ({a} − {b}) | 95% CI (t) | wins/ties/losses | Wilcoxon p | Holm p |",
             "|---|---|---|---|---|---|---|"]
    cells, pvalues = [], []
    for n in sorted(deltas["n"].unique()):
        for dataset in DATASETS:
            x = deltas[(deltas.n == n) & (deltas.dataset == dataset)]["delta"].to_numpy()
            if len(x):
                cells.append((n, dataset, x))
                pvalues.append(wilcoxon_p(x))
    finite = [p for p in pvalues if not np.isnan(p)]
    adjusted = iter(holm(finite)) if finite else iter([])
    for (n, dataset, x), p in zip(cells, pvalues):
        lo, hi = t_ci(x)
        wtl = f"{(x > 0).sum()}/{(x == 0).sum()}/{(x < 0).sum()}"
        p_holm = next(adjusted) if not np.isnan(p) else np.nan
        lines.append(f"| {n} | {dataset} | {x.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {wtl} | {p:.3g} | {p_holm:.3g} |")
    x = deltas["delta"].to_numpy()
    lo, hi = stratified_bootstrap_ci(deltas)
    pooled = {"mean": x.mean(), "ci": (lo, hi), "p": wilcoxon_p(x), "n_pairs": len(x),
              "wins": int((x > 0).sum()), "losses": int((x < 0).sum())}
    lines.append(f"| all | **pooled ({len(x)} pairs)** | **{x.mean():+.2f}** | [{lo:+.2f}, {hi:+.2f}] (bootstrap) | "
                 f"{(x > 0).sum()}/{(x == 0).sum()}/{(x < 0).sum()} | {pooled['p']:.3g} | |")
    return "\n".join(lines) + "\n", pooled


def score_table(df: pd.DataFrame, losses: list, with_paper: bool) -> str:
    header = "| N | dataset | " + " | ".join(losses) + (" | paper (Cosine) |" if with_paper else " |")
    lines = [header, "|" + "---|" * (len(losses) + 2 + with_paper)]
    for n in sorted(df["n"].unique()):
        for dataset in DATASETS:
            cells = []
            for loss in losses:
                s = df[(df.n == n) & (df.dataset == dataset) & (df.loss == loss)]["score"]
                cells.append(f"{fmt(s.mean(), s.std(ddof=1))} ({len(s)})" if len(s) else "–")
            paper = PAPER.get(n, {}).get(dataset)
            tail = f" | {fmt(*paper)} |" if with_paper and paper else (" | – |" if with_paper else " |")
            lines.append(f"| {n} | {dataset} | " + " | ".join(cells) + tail)
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load(RUNS_DIR)
    if df.empty:
        sys.exit(f"no runs in {RUNS_DIR}")
    df.to_csv(HERE / "runs.csv", index=False)
    out = ["# SINCERE vs SupCon vs Cosine in SetFit: results\n",
           f"{len(df)} runs, git commits: {', '.join(sorted(df.git_commit.str[:7].unique()))}. "
           "Scores ×100: accuracy, except amazon_counterfactual_en (Matthews correlation), as in the SetFit paper. "
           "Cells: mean ± std over splits (number of splits).\n"]

    main_runs = df[df.experiment.isin(["main-n8", "main-n64"])]
    out += ["## 1. Main results (bf16, temperature 0.07, step-matched)\n",
            score_table(main_runs, ["cosine", "supcon", "sincere"], with_paper=True)]
    for a, b in [("sincere", "supcon"), ("sincere", "cosine"), ("supcon", "cosine")]:
        table, _ = comparison_table(main_runs, a, b)
        out += [f"### {a} vs {b}\n", table]

    default = df[df.experiment == "default-budget"]
    if not default.empty:
        out += ["## 2. SetFit's default budget for SupCon (1 epoch, batch 16)\n",
                score_table(default, ["supcon", "sincere"], with_paper=False),
                comparison_table(default, "sincere", "supcon")[0]]

    out += ["## 3. Temperature sensitivity (N=8)\n",
            "| temperature | SupCon mean | SINCERE mean | Δ (SINCERE − SupCon) | 95% CI (bootstrap) | wins/losses | Wilcoxon p |",
            "|---|---|---|---|---|---|---|"]
    for name, t in [("main-n8", 0.07), ("temp-0.1", 0.1), ("temp-0.3", 0.3), ("temp-0.5", 0.5)]:
        sub = df[df.experiment == name]
        _, pooled = comparison_table(sub, "sincere", "supcon")
        if pooled:
            out.append(f"| {t} | {sub[sub.loss == 'supcon'].score.mean():.2f} | {sub[sub.loss == 'sincere'].score.mean():.2f} "
                       f"| {pooled['mean']:+.2f} | [{pooled['ci'][0]:+.2f}, {pooled['ci'][1]:+.2f}] "
                       f"| {pooled['wins']}/{pooled['losses']} | {pooled['p']:.3g} |")
    out.append("")

    parity = df[df.experiment == "parity-fp32"]
    if not parity.empty:
        out += ["## 4. Precision check: fp32 rerun vs bf16 (N=8, splits 0–4)\n",
                "| loss | mean abs(fp32 − bf16) | max abs(fp32 − bf16) | mean fp32 − bf16 |", "|---|---|---|---|"]
        base = df[df.experiment == "main-n8"].set_index(["loss", "dataset", "split"])["score"]
        joined = parity.set_index(["loss", "dataset", "split"])["score"].to_frame("fp32").join(base.rename("bf16"), how="inner")
        for loss, g in joined.groupby(level="loss"):
            d = g.fp32 - g.bf16
            out.append(f"| {loss} | {d.abs().mean():.2f} | {d.abs().max():.2f} | {d.mean():+.2f} |")
        fp32_delta, _ = comparison_table(parity, "sincere", "supcon")
        out += ["", "SINCERE − SupCon measured in fp32 only:\n", fp32_delta]

    noise = df[df.experiment == "noise-repeat"]
    if not noise.empty:
        out += ["## 5. Run-to-run noise (identical configuration and seed, rerun)\n",
                "| loss | mean abs(rerun − original) | max abs(rerun − original) |", "|---|---|---|"]
        base = df[df.experiment == "main-n8"].set_index(["loss", "dataset", "split"])["score"]
        joined = noise.set_index(["loss", "dataset", "split"])["score"].to_frame("rerun").join(base.rename("orig"), how="inner")
        for loss, g in joined.groupby(level="loss"):
            d = (g.rerun - g.orig).abs()
            out.append(f"| {loss} | {d.mean():.2f} | {d.max():.2f} |")
        out.append("")

    out += ["## 6. Compute\n", "| experiment | runs | train hours | eval hours |", "|---|---|---|---|"]
    for name, g in df.groupby("experiment", sort=False):
        out.append(f"| {name} | {len(g)} | {g.train_seconds.sum() / 3600:.2f} | {g.eval_seconds.sum() / 3600:.2f} |")
    out.append(f"| **total** | {len(df)} | {df.train_seconds.sum() / 3600:.2f} | {df.eval_seconds.sum() / 3600:.2f} |\n")

    (HERE / "RESULTS.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
