"""Aggregate results/seed*.json into RESULTS.md (the chart's table layout) and results/jev_chart_setfit.png.

    uv run --no-project --with matplotlib python report.py

Published columns are parsed from upstream report/EVALUATION_REPORT.md ("Accuracy results" table).
SetFit = mean ± sample std over seeds of jsb F1 at threshold 0.5.
Gold-label runs (results/gold/<model>_k<shots>/seed*.json) get their own RESULTS.md section, never the chart.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT = HERE / "upstream" / "report" / "EVALUATION_REPORT.md"

# chart order: (task id, upstream table key, row label, group title, group subtitle)
ROWS = [
    ("wg_prompt_harmful", "wildguardtest/wg_prompt_harmful", "WildGuardTest prompt", "WildGuardTest", "PROMPT"),
    ("ag2_unsafe_prompt", "aegis_v2/ag2_unsafe_prompt", "Aegis v2 prompt", "Aegis v2", "PROMPT"),
    ("tc_toxic", "toxicchat/tc_toxic", "ToxicChat prompt", "ToxicChat", "PROMPT"),
    ("hb_harmful_response", "harmbench/hb_harmful_response", "HarmBench response", "HarmBench", "RESPONSE"),
    ("ag2_unsafe_response", "aegis_v2/ag2_unsafe_response", "Aegis v2 response", "Aegis v2", "RESPONSE"),
]
MODELS = ["Jev", "Shieldstral-1.0-3B", "Qwen3Guard-8B", "GPT-OSS-Safeguard-20B", "Nemotron-3.5-Safety-4B"]
REPORT_COLS = ["Jev F1", "Shieldstral announced", "Qwen3Guard-8B", "GPT-OSS-Safeguard-20B", "Nemotron-3.5-Safety-4B"]
SETFIT = "SetFit (0 human labels)"
GO_F1 = 75.0  # gold-pilot go/no-go: some setting's seed-0 average F1 over the five tasks (README.md -> Status)


def published() -> dict[str, list[float]]:
    """task id -> [Jev, Shieldstral, Qwen3Guard, GPT-OSS, Nemotron] F1 (%) from the upstream report table."""
    lines = REPORT.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("| Task | Jev F1 |"))
    cols = [c.strip() for c in lines[head].strip("|").split("|")]
    idx = [cols.index(c) for c in REPORT_COLS]
    out = {}
    for line in lines[head + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        for tid, key, *_ in ROWS:
            if cells[0] == key:
                out[tid] = [float(re.sub(r"%$", "", cells[i])) for i in idx]
    assert set(out) == {r[0] for r in ROWS}, out.keys()
    return out


def load_seeds(d: Path = RESULTS) -> list[dict]:
    seeds = sorted(d.glob("seed*.json"), key=lambda p: int(p.stem[4:]))
    return [json.loads(p.read_text()) for p in seeds]


def gold_section() -> list[str]:
    """RESULTS.md lines for the gold-label runs: one row per (model, shots), F1 mean ± std over its seeds."""
    runs = []
    for d in (RESULTS / "gold").glob("*_k*"):
        seeds = load_seeds(d)
        if seeds:
            cfg = seeds[0]["config"]
            runs.append((cfg["model"], cfg["per_class"], seeds[0]["env"]["precision"], seeds, summarize(seeds)))
    if not runs:
        return []
    runs.sort(key=lambda r: (r[2], r[0], r[1]))
    seed0 = {(m, k, pr): 100 * statistics.mean(next(r for r in seeds if r["seed"] == 0)["tasks"][tid]["f1"] for tid, *_ in ROWS)
             for m, k, pr, seeds, _ in runs if any(r["seed"] == 0 for r in seeds)}
    best = max(seed0, key=seed0.get)
    hdr = ["Model", "per class", "training", "seeds"] + [label for _, _, label, *_ in ROWS] + ["Average"]
    lines = ["## Gold-label pilot (human labels; not the 0-label column)", "",
             "Same pools, test sets, hyperparameters and scoring as above, but the training examples are picked by their "
             "gold label instead of Opus's (pool shuffled with Random(seed), first k of each class, so the 8 are a subset of "
             "the 32, and the 32 of the 64). Purpose: check whether SetFit reaches the published range at all with clean labels. "
             "F1 (%), mean ± sample std over the listed seeds.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for m, k, pr, seeds, s in runs:
        cell = (lambda v: f"{v['mean']:.1f} ± {v['std']:.1f}") if len(seeds) > 1 else (lambda v: f"{v['mean']:.1f}")
        avg = cell(s["avg"])
        lines.append(f"| `{m.split('/')[-1]}` | {k} | {pr} | {', '.join(str(r['seed']) for r in seeds)} | "
                     + " | ".join(cell(s[tid]) for tid, *_ in ROWS) + f" | {'**' + avg + '**' if (m, k, pr) == best else avg} |")
    hb = sorted({n for *_, s in runs for n in s["hb_harmful_response"]["n"]}, reverse=True)
    verdict = "passes" if seed0[best] >= GO_F1 else "fails"
    lines += ["", f"HarmBench response excludes each run's training items from scoring (2 x per class), so its n is "
              f"{', '.join(map(str, hb))} across rows.", "",
              f"Go/no-go (written before the pilot ran): continue only if some setting reaches seed-0 average F1 >= {GO_F1:.0f}. "
              f"Best seed-0 setting: `{best[0].split('/')[-1]}` with {best[1]} per class ({best[2]} training), {seed0[best]:.1f}, "
              f"so the pilot {verdict}. bf16 rows are the first pilot; fp32 rows rerun it after bf16 was found to hurt mpnet "
              f"as training gets longer (README.md -> Status).", ""]
    return lines


def mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(seeds: list[dict]) -> dict:
    s = {}
    for tid, *_ in ROWS:
        f1 = [100 * r["tasks"][tid]["f1"] for r in seeds]
        half = [100 * (r["tasks"][tid]["f1_ci"][1] - r["tasks"][tid]["f1_ci"][0]) / 2 for r in seeds]
        m, sd = mean_std(f1)
        s[tid] = dict(f1=f1, mean=m, std=sd, ci_half=statistics.mean(half),
                      n=[r["tasks"][tid]["n"] for r in seeds],
                      train_s=[r["tasks"][tid]["train_seconds"] for r in seeds],
                      p50=[r["tasks"][tid]["latency"]["p50_ms"] for r in seeds],
                      p95=[r["tasks"][tid]["latency"]["p95_ms"] for r in seeds],
                      refused=[r["tasks"][tid]["n_refused"] for r in seeds],
                      labeled=[r["tasks"][tid]["n_labeled"] for r in seeds])
    avg = [statistics.mean(100 * r["tasks"][tid]["f1"] for tid, *_ in ROWS) for r in seeds]
    s["avg"] = dict(f1=avg, mean=mean_std(avg)[0], std=mean_std(avg)[1])
    return s


def write_markdown(pub: dict, s: dict, seeds: list[dict]) -> str:
    n = len(seeds)
    hdr = ["Task"] + MODELS + [SETFIT]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for tid, _, label, *_ in ROWS:
        v = s[tid]
        out.append(f"| {label} | " + " | ".join(f"{x:.1f}" for x in pub[tid]) + f" | {v['mean']:.1f} ± {v['std']:.1f} |")
    avg_pub = [statistics.mean(pub[tid][k] for tid, *_ in ROWS) for k in range(len(MODELS))]
    out.append("| **Average of all five tasks** | " + " | ".join(f"**{x:.1f}**" for x in avg_pub)
               + f" | **{s['avg']['mean']:.1f} ± {s['avg']['std']:.1f}** |")
    table = "\n".join(out)

    env = seeds[0]["env"]
    cfg = seeds[0]["config"]
    agree = json.loads((RESULTS / "opus_vs_gold.json").read_text()) if (RESULTS / "opus_vs_gold.json").exists() else {}
    all_train = [t for tid, *_ in ROWS for t in s[tid]["train_s"]]
    all_p50 = [t for tid, *_ in ROWS for t in s[tid]["p50"]]
    all_p95 = [t for tid, *_ in ROWS for t in s[tid]["p95"]]
    hb_n = sorted(set(s["hb_harmful_response"]["n"]))
    lines = [
        "# SetFit on the Jev safety benchmark (0 human labels)",
        "",
        *([f"> **Work in progress: {n} of 3 planned seeds.** Seed-to-seed std is not yet measured "
           f"(shown as ± 0.0); see README.md → Status for what is left.", ""] if n < 3 else []),
        f"F1 (%) per task. Published columns are copied from upstream "
        f"[`report/EVALUATION_REPORT.md`](https://github.com/mbburabak/jev-safety-benchmark/blob/1bf1eac/report/EVALUATION_REPORT.md) "
        f"(\"Accuracy results\" table; Jev rounded to one decimal as in the LinkedIn chart). "
        f"SetFit is mean ± sample std over {n} seed{'s' if n != 1 else ''}; each seed draws a new pseudo-label sample and a new training seed.",
        "",
        table,
        "",
        "![chart](results/jev_chart_setfit.png)",
        "",
        "## Setup",
        "",
        f"- **Labels per class:** {cfg['per_class']} positive + {cfg['per_class']} negative training examples per task and seed, "
        "labeled by Opus 4.8 (`claude -p --model claude-opus-4-8`, Pro plan). No human labels are used for training or selection.",
        f"- **Model:** `{cfg['model']}`, {seeds[0]['n_params_body'] / 1e6:.1f}M parameters in the body "
        "(counted on the trained model) plus a logistic-regression head on 768 dims.",
        f"- **Training:** SetFit, batch size {cfg['batch_size']}, max_length {cfg['max_length']}, {cfg['num_iterations']} "
        "pair iterations, 1 body epoch; prediction positive when P(positive) >= 0.5 (upstream's threshold).",
        f"- **Hardware:** Colab {env['gpu']} ({env['precision']} training, fp32 inference), torch {env['torch']}, setfit {env['setfit']}.",
        f"- **Training time:** {statistics.mean(all_train):.1f} s per task on average "
        f"(range {min(all_train):.1f}-{max(all_train):.1f} s), excluding model download.",
        f"- **Latency (GPU, batch 1, 100 test items per task):** p50 {statistics.median(all_p50):.1f} ms, "
        f"p95 {statistics.median(all_p95):.1f} ms (median across tasks and seeds). Jev's published single-flight p50 is 493.8 ms "
        "(upstream report, \"Latency headline\"), measured over the network; SetFit's is local GPU time, so the two are not like for like.",
        f"- **Test sets:** loaded through upstream `jsb.datasets` loaders; metrics from `jsb.metrics.stats` (F1 with Wilson CI). "
        f"WildGuardTest from the public walledai mirror (1,699 prompts), as in the published Jev run. "
        f"HarmBench response pool is the test set itself, so each seed's 16 training items are excluded from scoring: n = {', '.join(map(str, hb_n))} (published n 596).",
        "",
        "## Seeds vs. confidence intervals",
        "",
        "| Task | F1 per seed | std | mean Wilson 95% CI half-width |",
        "|---|---|---|---|",
    ]
    for tid, _, label, *_ in ROWS:
        v = s[tid]
        lines.append(f"| {label} | {', '.join(f'{x:.1f}' for x in v['f1'])} | {v['std']:.1f} | {v['ci_half']:.1f} |")
    wide = [label for tid, _, label, *_ in ROWS if s[tid]["std"] > s[tid]["ci_half"]]
    verdict = ("Not applicable yet: std needs at least 2 seeds." if n < 2 else
               f"Still wider: {', '.join(wide)}." if wide else "No task is wider; no more seeds needed.")
    lines += ["", "Rule: add seeds while a task's seed-to-seed std is wider than its Wilson CI half-width. " + verdict, ""]
    lines += ["## Opus 4.8 vs. gold on the labeled pool items", "",
              "Computed after selection; gold labels were never used to select training examples.", "",
              "| Task | items sent | refused | with gold | agreement | Opus+ / gold+ | Opus+ / gold- | Opus- / gold+ |",
              "|---|---|---|---|---|---|---|---|"]
    for tid, _, label, *_ in ROWS:
        a = agree.get(tid)
        if a:
            lines.append(f"| {label} | {a['n_sent']} | {a['refused']} | {a['n']} | {100 * a['rate']:.1f}% | {a['tp']} | {a['fp']} | {a['fn']} |")
    lines += ["",
              "Refused = blocked under the usage policy even when the batch was split down to that single item "
              "(mostly chemical/biological HarmBench generations); such items are never used for training but stay in scoring.",
              "",
              *gold_section(),
              *auprc_section(),
              "## Caveats",
              "",
              "- These public datasets (WildGuardMix, Aegis 2.0, ToxicChat, HarmBench) predate the training cutoffs of both Opus and Jev, "
              "so either may have seen them during training; the comparison does not control for contamination.",
              "- ToxicChat pool: the 0124 train split filtered to `human_annotation == True`, the same filter upstream applies to its test split.",
              "- The labeler is Opus 4.8, not Opus 5.5: with `--model opus`, Opus 5.5's safeguards stopped the first batch and Claude Code fell back to Opus 4.8.",
              "- Published numbers for the four guard models come from the Shieldstral paper and model card via the upstream report; "
              "they were not re-run here, and no paired test against them is possible.",
              ""]
    md = "\n".join(lines)
    (HERE / "RESULTS.md").write_text(md)
    return table


# ---------------------------------------------------------------- AUPRC (threshold-free)
DATA = HERE / "upstream" / "data"
BENCH = {tid: key.split("/")[0] for tid, key, *_ in ROWS}


def auprc(y: list[bool], p: list[float]) -> float:
    """Average precision (sklearn's step-wise AUPRC): sum over distinct scores, high to low, of recall gain x precision."""
    pairs = sorted(zip(p, y), key=lambda t: -t[0])
    n_pos, tp, seen, prev_r, ap, i = sum(y), 0, 0, 0.0, 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:  # a tie group is one threshold
            tp += pairs[j][1]
            j += 1
        seen += j - i
        r = tp / n_pos
        ap += (r - prev_r) * tp / seen
        prev_r, i = r, j
    return ap


def banded(p: list[float]) -> list[int]:
    """Upstream's 0.1 reliability bands ([0.0, 0.1) ... [0.9, 1.0]) as tied scores 0..9."""
    return [sum(x >= k / 10 for k in range(1, 10)) for x in p]


def gold_by_case() -> dict[str, dict[str, bool]]:
    """task id -> case id -> gold label, from upstream's local case files (the same rows tasks.load_test reads)."""
    out = {tid: {} for tid in BENCH}
    for bench in set(BENCH.values()):
        for line in open(DATA / f"{bench}_cases.jsonl"):
            row = json.loads(line)
            for tid in row["tasks"]:
                if tid in out and row["gold"].get(tid) is not None:
                    out[tid][row["case_id"]] = bool(row["gold"][tid])
    return out


def setfit_auprc(pred_file: Path, gold: dict) -> dict[str, tuple[float, float]]:
    """task id -> (AUPRC on exact scores, AUPRC on the same scores put in upstream's 10 bands)."""
    preds = json.loads(pred_file.read_text())
    out = {}
    for tid, by_case in preds.items():
        y = [gold[tid][c] for c in by_case]
        p = list(by_case.values())
        out[tid] = (auprc(y, p), auprc(y, banded(p)))
    return out


def jev_auprc() -> dict[str, tuple[float, float]]:
    """task id -> (min, max) AUPRC of Jev from upstream's 10-band reliability tables, items in a band tied.

    A band's positive count is n x rate, but rates are rounded to 3 decimals, so a large band can fit 2+ counts; only
    combinations that reproduce the report's total positives (tp + fn) and its tp at 0.5 are kept.
    """
    import itertools

    text = REPORT.read_text()
    out = {}
    for tid, key, *_ in ROWS:
        head = re.search(rf"#### {tid}\n\n- n=(\d+) \(tp=(\d+) fp=\d+ tn=\d+ fn=(\d+)\)", text)
        n_all, tp, fn = map(int, head.groups())
        sec = text.split(f"### {key}\n", 1)[1].split("\n### ", 1)[0]
        bands = [(int(float(lo) * 10), int(n), float(rate)) for lo, n, rate in
                 re.findall(r"\| \[(\d\.\d), \d\.\d[)\]] \| (\d+) \| [\d.]+ \| ([\d.]+) \|", sec)]
        assert sum(n for _, n, _ in bands) == n_all, (tid, bands)
        cands = [[k for k in range(n + 1) if abs(k / n - rate) <= 0.0005 + 1e-12] for _, n, rate in bands]
        vals = []
        for ks in itertools.product(*cands):
            if sum(ks) != tp + fn or sum(k for (b, _, _), k in zip(bands, ks) if b >= 5) != tp:
                continue
            y, p = [], []
            for (b, n, _), k in zip(bands, ks):
                y += [True] * k + [False] * (n - k)
                p += [b] * n
            vals.append(auprc(y, p))
        assert vals, f"{tid}: no band counts reproduce tp={tp}, fn={fn}"
        out[tid] = (min(vals), max(vals))
    return out


def auprc_section() -> list[str]:
    """RESULTS.md lines: AUPRC of Jev (from its reliability tables) and of every SetFit run with stored predictions."""
    gold = gold_by_case()
    jev = jev_auprc()
    pos_rate = {tid: statistics.mean(gold[tid].values()) for tid in BENCH}
    zero = [setfit_auprc(f, gold) for f in sorted((RESULTS / "predictions").glob("seed*.json"))]
    fmt = lambda lo, hi: f"{100 * lo:.1f}" if hi - lo < 0.0005 else f"{100 * lo:.1f}-{100 * hi:.1f}"
    lines = ["## AUPRC (threshold-free)", "",
             "Area under the precision-recall curve, as average precision (sklearn's definition), on the positive class. "
             "A random ranking scores the positive rate. Upstream does not publish Jev's per-case probabilities, only a "
             "reliability table per task (count and positive rate in each 0.1 probability band), so Jev's AUPRC treats all "
             "items in a band as tied; SetFit is shown both on its exact scores and put in the same 10 bands, which is the "
             "like-for-like comparison. A range means more than one set of band counts fits the report's rounded rates.", "",
             f"| Task | positive rate | Jev (10 bands) | SetFit 0 labels, 10 bands | SetFit 0 labels, exact |", "|---|---|---|---|---|"]
    for tid, _, label, *_ in ROWS:
        ex = [z[tid][0] for z in zero]
        bd = [z[tid][1] for z in zero]
        sd = lambda xs: f" ± {100 * statistics.stdev(xs):.1f}" if len(xs) > 1 else ""
        lines.append(f"| {label} | {100 * pos_rate[tid]:.1f} | {fmt(*jev[tid])} | {100 * statistics.mean(bd):.1f}{sd(bd)} | "
                     f"{100 * statistics.mean(ex):.1f}{sd(ex)} |")
    avg = lambda xs: 100 * statistics.mean(xs)
    lines.append(f"| **Average of all five tasks** | {avg(pos_rate.values()):.1f} | "
                 f"**{fmt(avg([jev[t][0] for t in BENCH]) / 100, avg([jev[t][1] for t in BENCH]) / 100)}** | "
                 f"**{statistics.mean(avg([z[t][1] for t in BENCH]) for z in zero):.1f}** | "
                 f"**{statistics.mean(avg([z[t][0] for t in BENCH]) for z in zero):.1f}** |")
    lines += ["", f"SetFit 0 labels: mean over {len(zero)} seed{'s' if len(zero) != 1 else ''}. Positive rate is over upstream's "
              "full test set; SetFit's HarmBench response scores exclude each seed's training items.", ""]
    runs = []
    for d in sorted((RESULTS / "gold").glob("*_k*")):
        for f in sorted((d / "predictions").glob("seed0.json")):
            cfg = json.loads((d / "seed0.json").read_text())
            runs.append((cfg["config"]["model"].split("/")[-1], cfg["config"]["per_class"], cfg["env"]["precision"],
                         setfit_auprc(f, gold)))
    if runs:
        runs.sort(key=lambda r: (r[2], r[0], r[1]))
        lines += ["Gold-label runs (seed 0, exact scores):", "",
                  "| Model | per class | training | " + " | ".join(label for _, _, label, *_ in ROWS) + " | Average |",
                  "|" + "---|" * (4 + len(ROWS))]
        for m, k, pr, a in runs:
            lines.append(f"| `{m}` | {k} | {pr} | " + " | ".join(f"{100 * a[tid][0]:.1f}" for tid in BENCH)
                         + f" | {avg([a[tid][0] for tid in BENCH]):.1f} |")
        lines.append("")
    return lines


# ---------------------------------------------------------------- chart (recreates the LinkedIn post's layout)
BG = "#F2F2F2"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
RED = "#E8321E"
RED_TEXT = "#B5260F"
COLORS = [RED, "#2B2B2B", "#5E5C5A", "#9E9B98", "#D9D6D4", "#1F5FD6"]
LEGEND = ["Jev", "Shieldstral-1.0-3B", "Qwen3Guard-8B", "GPT-OSS-Safeguard-20B", "Nemotron-3.5-Safety-4B", SETFIT]
SHORT = ["Jev", "Shieldstral", "Qwen3Guard", "GPT-OSS", "Nemotron-3.5", "SetFit"]
W, H = 1200, 610  # px at 1x; saved at 2x


def make_chart(pub: dict, s: dict, n_seeds: int, n_params: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    checked = []  # every text artist, for the overflow / overlap check

    def text(x, y, s_, size, color=INK, weight="normal", family="Helvetica Neue", ha="left", va="baseline", **kw):
        t = ax.text(x, y, s_, fontsize=size * 0.75, color=color, fontweight=weight, family=family, ha=ha, va=va, **kw)
        checked.append(t)
        return t

    def width(t):
        fig.canvas.draw() if not hasattr(fig.canvas, "_drawn") else None
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        return bb.width

    L, R = 36, W - 36
    text(L, 58, "Jev vs. four SOTA guard models + SetFit", 33, family="Arial Black", weight="black")
    text(R, 56, "SAFETY CLASSIFICATION · F1 (%)", 15, color=RED, weight="bold", ha="right")
    ax.plot([L, R], [80, 80], color=INK, lw=2)

    # legend row
    x = L
    for k, name in enumerate(LEGEND):
        ax.add_patch(Rectangle((x, 101), 15, 15, facecolor=COLORS[k], edgecolor=INK if k == 4 else "none", lw=1.2))
        t = text(x + 23, 115, name, 15, color=RED_TEXT if k == 0 else INK, weight="bold" if k in (0, 5) else "normal")
        x += 23 + width(t) + 22
    legend_end = x - 22

    # bar groups
    base, scale = 360, 1.83  # px per F1 point, as in the post (88.4 -> ~162 px)
    gw, gap = 204, 27
    bw, bgap = 29, 5
    for g, (tid, _, _, title, sub) in enumerate(ROWS):
        gx = L + g * (gw + gap)
        vals = pub[tid] + [s[tid]["mean"]]
        best = max(round(v, 1) for v in vals)
        for k, v in enumerate(vals):
            bx = gx + k * (bw + bgap)
            h = v * scale
            ax.add_patch(Rectangle((bx, base - h), bw, h, facecolor=COLORS[k],
                                   edgecolor=INK if k == 4 else "none", lw=1.6 if k == 4 else 0))
            text(bx + bw / 2, base - h - 7, f"{v:.1f}", 12.5, color=RED_TEXT if k == 0 else INK,
                 weight="bold" if round(v, 1) == best else "normal", ha="center")
        ax.plot([gx, gx + 6 * bw + 5 * bgap], [base + 6, base + 6], color=INK, lw=2.4)
        text(gx, base + 34, title, 19, weight="bold")
        text(gx, base + 59, sub, 11.5, color=MUTED, weight="bold")
        text(gx, base + 79, "CLASSIFICATION", 11.5, color=MUTED, weight="bold")

    # average row
    ax.plot([L, R], [478, 478], color="#9A9A9A", lw=1.2)
    avg = [statistics.mean(pub[tid][k] for tid, *_ in ROWS) for k in range(5)] + [s["avg"]["mean"]]
    x = L
    t = text(x, 512, "AVERAGE OF ALL FIVE TASKS", 13, color=MUTED, weight="bold")
    x += width(t) + 20
    for k, v in enumerate(avg):
        t = text(x, 513, f"{v:.1f}", 19, color=RED_TEXT if k == 0 else INK, weight="bold")
        x += width(t) + 5
        t = text(x, 512, SHORT[k], 13, color=MUTED if k else RED_TEXT)
        x += width(t) + 18
    avg_end = x - 18

    text(L, 556, f"SetFit: all-mpnet-base-v2 ({n_params / 1e6:.0f}M parameters), 8 examples per class labeled by Opus 4.8, no human labels; "
                 f"mean of {n_seeds} seed{'s' if n_seeds != 1 else ''} (± std in the table).", 12, color=MUTED)
    text(L, 578, "Other models: published F1 from the jev-safety-benchmark report (Shieldstral paper and model card). "
                 "WildGuardTest via the public mirror.", 12, color=MUTED)

    # overflow / overlap check on every text artist
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(t, t.get_window_extent(r)) for t in checked]
    problems = []
    for t, bb in boxes:
        if bb.x0 < L - 1 or bb.x1 > R + 1 or bb.y0 < 0 or bb.y1 > H:
            problems.append(f"overflow: {t.get_text()!r} x={bb.x0:.0f}-{bb.x1:.0f}")
    for i, (t1, b1) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if b1.overlaps(b2):
                problems.append(f"overlap: {t1.get_text()!r} / {t2.get_text()!r}")
    if legend_end > R or avg_end > R:
        problems.append(f"row too wide: legend ends {legend_end:.0f}, average ends {avg_end:.0f}, limit {R}")
    for p in problems:
        print("MISMATCH", p)
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out} ({len(problems)} layout problems)")


def main():
    pub = published()
    seeds = load_seeds()
    s = summarize(seeds)
    print(write_markdown(pub, s, seeds))
    make_chart(pub, s, len(seeds), seeds[0]["n_params_body"], RESULTS / "jev_chart_setfit.png")


if __name__ == "__main__":
    main()
