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
