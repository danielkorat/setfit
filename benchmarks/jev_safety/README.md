# SetFit on the Jev safety benchmark (0 human labels)

Adds a "SetFit (0 human labels)" column to the "Jev vs. four SOTA guard models" chart (F1 %) from
[mbburabak/jev-safety-benchmark](https://github.com/mbburabak/jev-safety-benchmark) at commit `1bf1eac`,
using that harness's own test loaders, gold labels, threshold (0.5) and metrics (`jsb.metrics.stats`, F1 with Wilson CI).

Results: [RESULTS.md](RESULTS.md) (table, setup, seeds, Opus-vs-gold agreement, caveats) and
[results/jev_chart_setfit.png](results/jev_chart_setfit.png).

## Status (3 Oct 2026, work in progress)

- **Done:** seed 0 on all five tasks (Colab L4, bf16): average F1 59.8 vs. 82.6-86.5 for the published models.
  Opus 4.8 labels agree with gold on 82.8-98.6% of labeled pool items, and seed 0's training labels match gold on
  14-16 of 16 per task, so label noise is not the main cause of the gap.
- **Not done:** seeds 1 and 2 (seed 1 has 3 of 5 tasks labeled in `results/opus_labels/`; selections live in the
  gitignored `work/`), the seed-vs-CI rule, the final chart.
- **Cost warning:** each `claude -p` call writes ~25k tokens of Claude Code's own context to cache, and calls refused
  by safeguards bill like successful ones. The labeling loop (165 calls, 53 refused) emptied three Pro 5-hour windows.
  Before relabeling, cut per-call overhead (one call per 64 items, no CLAUDE.md/memory loading) and drop retries on
  refusals.
- **Gold-label pilot (3 Oct 2026, written before the pilot ran):** to check whether SetFit can reach the published
  range at all before spending more on labeling, train on gold labels (no `claude -p` calls), seed 0, 8/32/64 examples
  per class, `all-mpnet-base-v2` and `BAAI/bge-base-en-v1.5` (`run_colab.py --labels gold --model ... --shots ...`).
  Go/no-go: continue only if some setting reaches average F1 >= 75 over the five tasks; if so, run seeds 0-2 for the
  best setting. These runs use human labels and are reported apart from the "0 human labels" column.
  **Result: fails.** Best is `bge-base-en-v1.5` with 64 per class, average F1 69.8; mpnet gets worse with more
  examples (60.0 / 53.1 / 56.1 at 8 / 32 / 64). Table: [RESULTS.md](RESULTS.md) → Gold-label pilot. Seeds 1-2 not run.
- **ToxicChat search (3 Oct 2026, written before stage 1 ran):** a colleague reports 80.6 on ToxicChat with SetFit's
  zero-shot recipe (`get_templated_dataset`); which metric that is (toxic-class F1, macro F1 or accuracy) is being
  asked. `search_tc.py` stage 1: 5 models x {3 zero-shot label pairs, gold 8 and 32 per class, gold 8 + zero-shot},
  seed 0, scored on a 1,000-item validation split from the train pool and on the test set. Go/no-go for stage 2
  (refining around the best setting): some setting reaches validation toxic-class F1 >= 70 (at 0.5 or at the
  validation-tuned threshold).
  **Result: fails, stage 2 not run.** Best validation toxic F1 is 66.4 (`all-mpnet-base-v2`, gold 8 per class,
  validation-tuned threshold; test 57.2); best test F1 at a validation-tuned threshold is 63.1 (`bge-base-en-v1.5`,
  gold 32 per class). Every zero-shot setting stays at or below 39.1 on validation and 30.7 on test. Few-shot
  rankings are decent (test AUROC 84-90) but at 0.5 the models flag too many prompts (13% of test prompts are toxic).
  Per-run numbers: `results/search/tc_stage1.json`.
  **The colleague's recipe** (`--stage moshe`: `BAAI/bge-base-en`, labels "neutral"/"toxic", 8 per label,
  batch 16, `max_steps=30`, seeds 0-2) reaches accuracy 58.3-61.9 on all 5,082 rows of the 0124 train split
  (always answering "not toxic" scores 92.4 there) and test toxic F1 27.6-28.9; with `bge-base-en-v1.5`, 77.0-79.5
  and 31.8-32.2. His snippet scores `metric="accuracy"`, so 80.6 is most likely accuracy, not the chart's F1.
  Differences from his run: bf16 autocast here, setfit at `a000da0`, and his evaluation split is not in the snippet.
  Per-run numbers: `results/search/tc_stagemoshe.json`.
- **Colab runs need the Mac awake:** a lid-closed sleep on battery drops `colab run`'s connection and the local
  process then waits forever (3 Oct 2026, two runs lost). Keep the lid open or the Mac on power.

## Method

1. **Test sets**: upstream `make download-data` with `JSB_HF_TOKEN` unset, so WildGuardTest loads from the public
   walledai mirror (1,699 prompts), as in the published Jev run. `tasks.py` asserts the published n per task.
2. **Unlabeled pools** (disjoint from the test sets, exact-text overlaps removed): Aegis 2.0 `train.json`
   (REDACTED rows dropped; rows with a response for the response task), ToxicChat 0124 train
   (`human_annotation == True`, the filter upstream applies to its test split), `allenai/wildguardmix` train (gated),
   and for HarmBench responses the 596 test items themselves (each seed's 16 training items are excluded from scoring).
3. **Pseudo-labels** (`label_opus.py`): per task and seed, 64 pool items (Random(seed)), labeled in batches of 8 by
   headless Claude Code on a Pro plan (`claude -p --model claude-opus-4-8 --output-format json --json-schema ...
   --permission-mode dontAsk`, from an empty temp dir). Each prompt carries the task's instructions, true/false criteria
   (`policy/question_sets.yaml`) and `policy/<bench>_policy.md`, the same content Jev received. More items are sampled
   until each class has 8; then 8 per class are picked. Gold labels are never used for selection.
   A batch blocked under the usage policy is halved until the blocked item is isolated; that item gets no label.
4. **SetFit on a Colab GPU** (`run_colab.py` + `colab_job.py`, `colab run --gpu L4`): `all-mpnet-base-v2`,
   8 examples per class, batch size 16, max_length 384; input is the prompt, or
   `"Response: {response}\n\nPrompt: {prompt}"` for response tasks; positive when P(positive) >= 0.5.
   Records batch-1 GPU latency on 100 test items per task.
5. **Seeds**: 0, 1, 2 (new pseudo-label sample and training seed each); more while a task's seed-to-seed std is
   wider than its Wilson CI half-width.

## Reproduce

```bash
git clone https://github.com/mbburabak/jev-safety-benchmark upstream && git -C upstream checkout 1bf1eac
(cd upstream && env -u JSB_HF_TOKEN make install && env -u JSB_HF_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 make download-data)
upstream/.venv/bin/python fetch_pools.py                 # needs `hf auth login` + accepted wildguardmix terms
upstream/.venv/bin/python label_opus.py --tasks all --seeds 0,1,2   # needs `claude auth login` (Pro)
upstream/.venv/bin/python label_opus.py --agreement
upstream/.venv/bin/python run_colab.py --seed 0          # needs the Colab CLI login; repeat per seed
uv run --no-project --with matplotlib python report.py
```

`upstream/`, `data/` and `work/` are gitignored; committed results contain case ids, labels and scores, never dataset text.
