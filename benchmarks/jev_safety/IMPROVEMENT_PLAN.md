# Raising SetFit F1 with at most 32 labels per class (plan, 3 Oct 2026)

**Question:** how to raise positive-class F1 of SetFit on the five Jev tasks with at most 32 labeled examples per
class, preferring setups that work at 8-16 (so a business can label the data). Baseline to beat: `all-mpnet-base-v2`,
32 gold labels per class, fp32 training, average F1 69.5 (seed 0, `results/gold/all-mpnet-base-v2_k32_fp32/`).
Jev averages 83.1 (upstream report).

**Sources:** this benchmark's own runs (gold pilot, `results/diag/mpnet_drop.json`, `results/search/`); the mteb source
([`abstasks/classification.py`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/classification.py),
[`toxic_conversations_classification.py`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/classification/eng/toxic_conversations_classification.py));
model cards linked below; the SetFit paper ([arXiv 2209.11055](https://arxiv.org/abs/2209.11055)); AncSetFit
([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.692.pdf)); Perez et al. on few-shot model selection
([arXiv 2105.11447](https://arxiv.org/abs/2105.11447)); huggingface/setfit PR #663.

## What the evidence already says

1. **The embedding model is the limit, not SetFit's body training.** A frozen body with only the logistic-regression
   head matched or beat full SetFit training at 32 per class (ToxicChat 59.4 vs 58.8, WildGuardTest 69.6 vs 66.7;
   `results/diag/mpnet_drop.json`).
2. **A better threshold alone cannot reach the published range.** Even the best threshold chosen on the test set
   itself adds about 3 points to the five-task average (69.5 -> 72.5 for mpnet 32 per class fp32; ToxicChat alone
   +7.5). Label-free EM prior correction (Saerens et al. 2002) failed: SetFit's probabilities are near 0 or 1, and the
   estimated positive rate collapsed (0% on ToxicChat for several runs).
3. **Our two baselines are retrieval-oriented embedders, which do poorly on toxicity classification.** MTEB's
   ToxicConversationsClassification is almost our setup: logistic regression on frozen embeddings, 16 examples per
   class, 10 repeats (mteb source above). Accuracy there, per the research summary from the mteb results repo:
   `jina-embeddings-v3` 91.3, `embeddinggemma-300m` 82.9, `Qwen3-Embedding-0.6B` 82.1, `gte-modernbert-base` 81.5,
   versus `bge-base-en-v1.5` 67.0 and `all-mpnet-base-v2` 61.1 [per-model numbers from a research summary of
   embeddings-benchmark/results; re-check each before quoting].
4. **Label noise is not the problem:** 8 Opus labels per class and 8 gold labels per class give the same average
   (59.8 vs 60.0, both bf16).
5. **Templated label-name examples hurt when added to real examples** in our search (bge-base, gold 8 per class:
   test F1 50.9 alone, 42.0 with the "not toxic"/"toxic" templates added; `results/search/tc_stage1.json`).
6. **Training settings:** bf16 autocast badly hurts mpnet in longer training (fp32 fixes it). The pinned SetFit
   commit (`a000da0`) never groups batches by label for SupCon and the triplet losses (PR #663, open), so those
   losses must use the PR's commit.

## Ranked steps

| # | Step | Why | Cost | Go/no-go (written before data) |
|---|---|---|---|---|
| 1 | **Embedder screen, frozen + logistic regression.** Embed every test set and the gold pools once per model on one Colab VM; fit logistic-regression heads for 8 / 16 / 32 per class x 5 seeds on the stored embeddings. Models: the two baselines, `Qwen3-Embedding-0.6B` (instruction prefix per task), `embeddinggemma-300m` (`task: classification \| query: `), `gte-modernbert-base`, `gte-large-en-v1.5`, `nomic-embed-text-v1.5` (`classification: `), `bge-large-en-v1.5`. | Evidence 1 + 3; one job tests all models at every label count, with 5 seeds instead of 1. | One L4 job, est. 30-45 min [estimate]. | Keep a model if its 5-seed mean five-task F1 at 32 per class beats frozen mpnet by >= 5 points. Continue to step 3 only if the best model reaches >= 72 at 32 per class or >= 69.5 (today's best) at 16. |
| 2 | **Threshold from the k labels, no extra labels.** Out-of-fold probabilities on the k-shot training set (cross-validation), plus the expected positive rate when the business knows it (flag the top share). Computed on step 1's stored scores, no new runs. | Evidence 2: worth up to +7 on ToxicChat, ~+1.5 on the average. | CPU only. | Adopt per method if it raises the 5-seed mean average by >= 1 point without lowering any task by > 2. |
| 3 | **SetFit fine-tuning on the best 1-2 embedders**, fp32, short training (1 epoch, `max_steps` cap), against their frozen version. | Fine-tuning may still help a stronger body; our evidence says it did not help mpnet. | One job. | Keep fine-tuning only if it beats frozen by >= 2 points on the 5-seed mean. |
| 4 | **Loss swap** (`SupConLoss`, `BatchHardTripletLoss`) on the step-3 winner, with SetFit at PR #663's commit. | No published few-shot comparison; low confidence. | One job. | Keep only if >= 2 points over the default loss. |
| 5 | **Self-training on the unlabeled pools** (pseudo-label with the step-3 model, retrain). | Pools are large and unlabeled data is free; risk of copying the over-flagging. | Medium. | Only if steps 1-3 leave a gap worth closing; rule to be written then. |

Not planned: AncSetFit-style anchors and templated examples (evidence 5 is against them for binary safety; AncSetFit's
gains shrink with k and were measured on multi-class accuracy); FastFit (gains reported on 50-150-class tasks only);
per-task hyperparameter search on a tiny validation set (Perez et al.: such selection is barely better than random).

## Search rules (all steps)

- Labels per class 8 / 16 / 32, seeds 0-4; report mean ± std over seeds, never one seed.
- Choose settings on the mean over all five tasks, not per task, and report the test numbers once.
- One Colab job per step; the Mac must stay awake (README → Status).
- Licenses matter for a business: `jina-embeddings-v3` / v5 are non-commercial (CC BY-NC), so they are left out of
  step 1 unless used only as a reference; `embeddinggemma-300m` is gated on Hugging Face (needs the license accepted
  and a token on the Colab VM).

## Results so far

- **Step 1 (embedder screen): fails both gates.** Best frozen model `Qwen3-Embedding-0.6B` (task instruction):
  64.2 / 66.0 / 68.0 average F1 at 8 / 16 / 32 per class, vs mpnet 58.2 / 62.0 / 65.6. Evidence 3 did not hold:
  `jina-embeddings-v3` and `gte-modernbert-base`, the MTEB toxicity leaders, score 65.1 and 64.9 at 32 per class.
  `embeddinggemma-300m` not run (job died on the VM, cause unknown). `results/screen/embed_*.json`.
- **Step 2 (threshold from the k labels): fails its rule.** +0.7 to +3.3 average for most models, but some task
  drops more than 2 points in most settings, and it hurts Qwen3 (ToxicChat -11.1 at 8 per class).
- Steps 3-5 not run.
