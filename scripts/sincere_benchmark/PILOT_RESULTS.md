# Does SINCERE improve SetFit? Pilot results

**Question.** Does replacing SetFit's embedding loss with SINCERE (Feeney & Hughes, arXiv 2309.14277) improve few-shot
classification accuracy over (a) SetFit's default `CosineSimilarityLoss` and (b) `SupConLoss`, the existing option
SINCERE corrects?

**Answer.** No measurable improvement over SetFit's default. Against Cosine, SINCERE averages −0.54 points and the
95% CI rules out a gain larger than about +0.3 to +0.5 points. Against SupCon it averages +0.40, with a CI that
includes 0. The full benchmark was therefore not run.

## Protocol

| Setting | Value |
|---|---|
| Model | `sentence-transformers/paraphrase-mpnet-base-v2`, max 256 tokens, logistic-regression head |
| Splits | SetFit's fixed few-shot splits (`setfit.utils.load_data_splits`, seeds 0–4) |
| Datasets, N=8 per class | emotion, sst5, SentEval-CR, amazon_counterfactual_en (the 4 short-text test sets of the SetFit paper) |
| Datasets, N=64 per class | SentEval-CR (binary: ~15 same-class samples per anchor per batch, where SupCon's intra-class repulsion is strongest) |
| Cosine budget | Paper protocol: 20 iterations, 16 pairs per step (32 sentences), 1 epoch |
| SupCon / SINCERE budget | Same number of steps as Cosine (2.5·N·C), up to 32 sentences per step, label-grouped batches |
| Temperature | 0.07 for both; SupCon's `base_temperature` set equal, so the denominator is the only difference |
| Precision, hardware | bf16 autocast on the transformer body, losses in fp32; Apple M5 GPU (MPS) |
| Code | branch `sincere-loss`, commits `9d4ad44` and `3e91bfd` (they differ only in `analyze.py`; `src/` is identical) |
| Metric | accuracy; Matthews correlation for amazon_counterfactual_en, as in the SetFit paper. Scores ×100 |

Go/no-go rule, fixed before the pilot ran: continue to the full benchmark only if (a) the pooled mean of
SINCERE − SupCon is above 0 and (b) the 95% CI of SINCERE − Cosine reaches 0 or above.

## Scores (mean ± std over 5 splits)

| N | Dataset | Cosine | SupCon | SINCERE | Paper, Cosine (10 splits) |
|---|---|---|---|---|---|
| 8 | sst5 | 43.2 ± 0.8 | 41.5 ± 2.4 | 40.8 ± 2.5 | 43.6 ± 3.0 |
| 8 | amazon_counterfactual_en | 36.3 ± 16.9 | 35.8 ± 15.4 | 37.3 ± 17.3 | 40.3 ± 11.8 |
| 8 | SentEval-CR | 89.6 ± 0.6 | 89.1 ± 1.4 | 89.2 ± 1.3 | 88.5 ± 1.9 |
| 8 | emotion | 50.4 ± 4.7 | 48.7 ± 3.7 | 49.6 ± 4.2 | 48.8 ± 4.5 |
| 64 | SentEval-CR | 90.5 ± 1.1 | 90.2 ± 0.3 | 90.4 ± 0.6 | 90.4 ± 0.6 |

Paper column: Tunstall et al., arXiv 2209.11055v1, Table 2 (SetFit-MPNet). Every Cosine cell is within one paper
standard deviation, so the setup reproduces the published baseline.

## Paired comparisons (25 pairs: same dataset, N and split)

| Comparison | Mean Δ | 95% CI (stratified bootstrap) | 95% CI (t) | Wins / ties / losses | Wilcoxon p |
|---|---|---|---|---|---|
| SINCERE − SupCon | +0.40 | [−0.14, +0.90] | [−0.25, +1.05] | 11 / 2 / 12 | 0.25 |
| SINCERE − Cosine | −0.54 | [−1.39, +0.29] | [−1.57, +0.49] | 9 / 0 / 16 | 0.24 |
| SupCon − Cosine | −0.94 | [−1.94, +0.08] | [−2.07, +0.20] | 9 / 1 / 15 | 0.07 |

Per cell, SINCERE − SupCon: sst5 −0.71, amazon_counterfactual_en +1.52, SentEval-CR (N=8) +0.05, emotion +0.90,
SentEval-CR (N=64) +0.21. No cell is significant after Holm correction (smallest Holm p = 0.31).

## Verdict

- The rule passed on both counts, but rule (b) was too lenient. It only asked that SINCERE not be clearly worse
  than Cosine. The data rules out a gain over Cosine larger than about +0.3 to +0.5 points at 95% confidence, and
  SINCERE loses 16 of 25 pairs. No useful improvement for SetFit is left within the confidence interval.
- SINCERE vs SupCon is unresolved: +0.40 on average, with a CI that includes 0. No effect appeared in the condition
  where the theory predicts the largest one (SentEval-CR, N=64: +0.21, CI [−0.70, +1.12]).
- This matches the SINCERE paper, which reports no significant difference from SupCon for linear-probe
  classification and gains only for transfer to new classes.

## Limits

- 5 splits, 4 datasets, mostly N=8. A gain smaller than about 1 point vs SupCon could be missed.
- enron_spam and ag_news (the long-text test sets) and N=64 on multi-class datasets were not run.
- Temperature fixed at 0.07; no sweep.

## Reproduce

```bash
cd scripts/sincere_benchmark
python run.py --only main-n8 --datasets emotion sst5 SentEval-CR amazon_counterfactual_en --limit-splits 5
python run.py --only main-n64 --datasets SentEval-CR --limit-splits 5
```

Per-run records (scores, step counts, timings, versions, git commit) are in `runs/`.
