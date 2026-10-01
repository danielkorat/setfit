# SetFit: current state as of October 2026 (codebase, limitations, demand, adoption, ecosystem position)

All metrics were collected on 2026-10-01 unless a different date is given. Local-repo facts come from `/home/user/setfit`, a shallow clone at commit `be332d6` (2026-09-23) with `setup.py` version `1.3.0.dev0`. Because the clone is shallow, only 54 commits back to Feb 2024 are visible locally. The GitHub page reports about 730 commits in total.

## 1. Architecture and code: module layout, design, dependency pins, and when feature work stopped

### Takeaway
SetFit is a small library, about 5.3k lines of Python in `src/setfit`. It sits on top of Sentence Transformers (ST). Since v1.1.0 (Sep 2024), the embedding phase runs through a backward-compatible subclass of `SentenceTransformerTrainer`. The classifier phase is still either an sklearn `LogisticRegression` saved with joblib, or a differentiable `SetFitHead` trained in a hand-written torch loop.

The last real feature work was v1.1.0 in Sep 2024. Every commit since then (Jan 2025 to Sep 2026) is compatibility work, CI or security hardening, or a typo fix. The most recent example is v1.2.0 (2026-09-04), which added compatibility with transformers v5, ST v6 and huggingface_hub v1.

### Cited Findings
**Module layout.** Line counts from `wc -l` on the local repo:
- `modeling.py` (918): `SetFitModel` and `SetFitHead`.
- `trainer.py` (901): `Trainer`, `BCSentenceTransformersTrainer`, the deprecated `SetFitTrainer`, and `ColumnMappingMixin`.
- `model_card.py` (580) plus a Jinja `model_card_template.md`.
- `training_args.py` (377).
- `span/` (about 760 in total): `AbsaModel`, `AspectModel`, `PolarityModel`, `SpanSetFitModel`, `AbsaTrainer` and a spaCy `AspectExtractor`.
- `logging.py` (344).
- `exporters/`: `onnx.py` (282), `openvino.py` (52), `utils.py`.
- `data.py` (273): few-shot sampling helpers and `get_templated_dataset` for zero-shot.
- `sampler.py` (184): `ContrastiveDataset` and `ContrastiveDistillationDataset`.
- `trainer_distillation.py` (164), `utils.py` (162), `losses.py` (100, `SupConLoss`), `notebook.py`, `integrations.py` (Optuna), and `compat.py` (44, version-shim imports).
- Total: 5,281 lines.
- Source: [repo src/setfit](https://github.com/huggingface/setfit/tree/main/src/setfit)

**Two-phase training design.**
- `TrainingArguments` is a SetFit-specific dataclass, not a subclass of HF `TrainingArguments`.
- Many of its fields are tuples that hold one value per phase:
  - `batch_size=(16, 2)`
  - `num_epochs=(1, 16)`
  - `body_learning_rate=(2e-5, 1e-5)`
  - `head_learning_rate=1e-2`
- Other defaults: `loss=CosineSimilarityLoss`, `margin=0.25`, `end_to_end=False`, `sampling_strategy="oversampling"`, `samples_per_label=2` (deprecated) and `metric_for_best_model="embedding_loss"`.
- Source: [training_args.py](https://github.com/huggingface/setfit/blob/main/src/setfit/training_args.py)

**Embedding phase delegated to ST.**
- `BCSentenceTransformersTrainer(SentenceTransformerTrainer)` translates SetFit args into ST args.
- It monkey-patches `callback_handler.call_event` so that callbacks receive the SetFit model and args.
- It removes ST's own model-card callback.
- When a batch-triplet loss or `SupConLoss` is used, it sets `BatchSamplers.GROUP_BY_LABEL`.
- This design came from PR #554 "Train via the Sentence Transformers Trainer from ST v3" (2024-09-18).
- Sources: [trainer.py](https://github.com/huggingface/setfit/blob/main/src/setfit/trainer.py); [commit fb91f67 / PR #554](https://github.com/huggingface/setfit/pull/554)

**Classifier phase.**
- `Trainer.train_classifier` calls `SetFitModel.fit()`.
- With the sklearn head, `fit()` encodes the inputs and calls `model_head.fit(embeddings, y)`.
- With the differentiable head, `fit()` runs its own loop: `AdamW` plus `StepLR(step_size=5, gamma=0.5)`, `trange`/`tqdm`, and manual `loss.backward()`.
- This loop does not use the HF/ST Trainer. It therefore has no distributed or multi-GPU support, no AMP integration, no logging callbacks and no checkpointing.
- `grep` finds no DataParallel, DDP or distributed code anywhere in `src/setfit`.
- Source: [modeling.py](https://github.com/huggingface/setfit/blob/main/src/setfit/modeling.py)

**Heads.**
- `SetFitHead` subclasses ST's `Dense` module.
- The default sklearn head is `LogisticRegression`.
- For multi-label, `multi_target_strategy` is one of `"one-vs-rest"` (`OneVsRestClassifier`), `"multi-output"` (`MultiOutputClassifier`) or `"classifier-chain"` (`ClassifierChain`).
- The sklearn head is serialized with `joblib.dump`/`joblib.load` (pickle) as `model_head.pkl`.
- Open issue #224 (12 comments): "This model has one file that has been marked as unsafe" on the HF Hub.
- Sources: [modeling.py](https://github.com/huggingface/setfit/blob/main/src/setfit/modeling.py); [issues sorted by comments](https://github.com/huggingface/setfit/issues?q=is%3Aissue+is%3Aopen+sort%3Acomments-desc)

**Pair sampling.**
- `ContrastiveDataset` is an `IterableDataset`. It enumerates every pair with `np.triu_indices` and shuffles them with a fixed `RandomState(seed=42)`.
- Strategies are "oversampling", "undersampling" and "unique", plus a legacy `num_iterations` option.
- Pair generation is materialized up front (`pos_pairs`/`neg_pairs` lists). This is the source of the out-of-memory problems on large datasets: issue #472 "Kernel crash due to out of memory for large dataset" (9 comments) and open PR #627 "Fix OOM in contrastive pair generation with streaming approach" (Jan 2026).
- Other open PRs target the sampler:
  - #646 "Exclude identity pairs"
  - #662 "Keep a one-example class from turning the contrastive loss into NaN" (2026-09-29)
  - #579 "Refactor ContrastiveDataset" (Dec 2024)
- Sources: [sampler.py](https://github.com/huggingface/setfit/blob/main/src/setfit/sampler.py); [open PRs](https://github.com/huggingface/setfit/pulls?q=is%3Apr+is%3Aopen+sort%3Areactions-desc)

**ABSA.**
- ABSA uses a spaCy pipeline (default `en_core_web_lg`) to extract noun-chunk aspect candidates.
- An `AspectModel` (a SetFit model) filters those candidates, and a `PolarityModel` (another SetFit model) classifies them.
- This needs the `absa` extra, which pins `spacy<3.8` and `numpy<2` on Python < 3.10.
- Sources: [span/modeling.py](https://github.com/huggingface/setfit/blob/main/src/setfit/span/modeling.py); [setup.py](https://github.com/huggingface/setfit/blob/main/setup.py)
- No other span or token-level tasks are implemented. There is no NER, regression or pair classification.

**Distillation.** `DistillationTrainer(Trainer)` trains a student SetFit model on the teacher's cosine-similarity targets over unlabeled text, using `ContrastiveDistillationDataset`. Source: [trainer_distillation.py](https://github.com/huggingface/setfit/blob/main/src/setfit/trainer_distillation.py)

**Exporters.**
- ONNX export wraps the body and head (`OnnxSetFitModel`) and calls `torch.onnx.export` with default `opset=12`.
- A code comment notes that "torch v2.9 made the dynamo based exporter the default, which does not support the opsets used here", so the code forces the legacy exporter.
- The sklearn head is exported through `skl2onnx`.
- OpenVINO export goes through ONNX plus `hummingbird-ml`. The setup.py comment says "hummingbird-ml pins onnx<=1.16.1, which has no wheels for Python 3.13".
- Related open bugs:
  - #291 "ONNX export for multilabel — discrepancy" (11 comments)
  - #355 "ONNX conversion of multi-output classifier" (8 comments)
  - #560 "Opset Issue while exporting to onnx"
  - #467 "Inconsistent onnx-optimum output"
- Sources: [exporters/onnx.py](https://github.com/huggingface/setfit/blob/main/src/setfit/exporters/onnx.py); [setup.py](https://github.com/huggingface/setfit/blob/main/setup.py); [issues](https://github.com/huggingface/setfit/issues?q=is%3Aissue+is%3Aopen+sort%3Acomments-desc)

**Model cards and Hub.**
- `SetFitModelCardData` and the Jinja template generate cards automatically. They cover dataset, ST body, head class, ABSA pipeline description, metrics, CodeCarbon emissions and library versions.
- `SetFitModel` uses huggingface_hub's `ModelHubMixin` (it was migrated away from `PyTorchModelHubMixin` in PR #505).
- `config_setfit.json` stores `normalize_embeddings`, `labels` and the span attributes.
- Sources: [model_card.py](https://github.com/huggingface/setfit/blob/main/src/setfit/model_card.py); [commit 3904e53 / PR #505](https://github.com/huggingface/setfit/pull/505)

**Zero-shot.** `get_templated_dataset()` builds synthetic examples from label names with the template "This sentence is {}". The docs have zero-shot tutorial and how-to pages. Sources: [data.py](https://github.com/huggingface/setfit/blob/main/src/setfit/data.py); [docs](https://huggingface.co/docs/setfit)

**Dependency pins (setup.py at be332d6, identical to PyPI 1.2.0).**
- Core: `datasets>=2.15.0`, `sentence-transformers[train]>=3`, `transformers>=4.41.0`, `evaluate>=0.4.6`, `huggingface_hub>=0.24.0`, `scikit-learn`, `packaging`, and `python_requires>=3.9`.
- The CI matrix includes a minimum-pins preset (`compat_tests`, which forces `pandas<2`, `fsspec<2023.12.0` and `numpy<2`) and a `compat_tests_v4` preset (`transformers<5`, `sentence-transformers<6`, `huggingface_hub<1`, `datasets<5`).
- The library therefore spans roughly three generations of the HF stack.
- Sources: [setup.py](https://github.com/huggingface/setfit/blob/main/setup.py); [PyPI JSON for setfit 1.2.0](https://pypi.org/pypi/setfit/json)

**Compatibility shims.**
- `compat.py` handles the module moves in ST v5.4 (`sentence_transformers.sentence_transformer.{losses,modules,training_args,model_card}`) and the transformers v5 relocation of `default_logdir`.
- `modeling.py` still contains branches for `sentence_transformers<2.3.0` (`use_auth_token`, `_target_device`), even though the minimum pin is ST 3. This is dead code.
- Code-level deprecations still pending "removal in v2.0.0": `SetFitTrainer`, `DistillationSetFitTrainer`, `Trainer.freeze/unfreeze`, `keep_body_frozen`, `evaluation_strategy` and `samples_per_label`.
- Sources: [compat.py](https://github.com/huggingface/setfit/blob/main/src/setfit/compat.py); [modeling.py](https://github.com/huggingface/setfit/blob/main/src/setfit/modeling.py)

**Commit history (local shallow log, Feb 2024 onward).**
Feature work in this window:
- Daniel Korat's PR #489 "Switch to differentiable head, larger input sequence" (2024-04-03).
- Optimum-Intel notebooks (#484, #517).
- PR #554, the ST v3 trainer delegation (2024-09-18).
- PR #557, the `NotebookCallback` (2024-09-19).

After v1.1.0 (2024-09-19), the commits are maintenance only:
- Compatibility fixes for transformers 4.45–4.50 (#577, #594), ST v4 (#595), datasets v4 (#614), and transformers v5 / ST v6 / hub v1 (#647).
- Typo fixes (#580, #589, #597).
- Doc-builder and CI work in Dec 2025 and Apr 2026.
- Security bot commits pinning action SHAs and scoping tokens (#631, #659, #660).
- Dependabot bumps.

Commit volume is low: 26 commits in 2024 (from Feb), 17 in 2025 and 11 in 2026. There were no commits at all between 2025-08-05 and 2025-12-11, or between 2026-04-15 and 2026-09-04.
- Sources: [commits](https://github.com/huggingface/setfit/commits/main); local `git log`

**Releases (PyPI).**

| Version | Date |
|---|---|
| 0.0.1 | 2022-07-08 |
| 0.1.0 | 2022-10-04 |
| 0.5.0 | 2022-12-14 |
| 0.6.0 | 2023-02-08 |
| 0.7.0 | 2023-04-14 |
| 1.0.0 | 2023-12-06 (new Trainer/TrainingArguments, ABSA, evaluation during training, early stopping) |
| 1.0.3 | 2024-01-16 |
| 1.1.0 | 2024-09-19 (defers embedding fine-tuning to ST, drops Python 3.7) |
| 1.1.1 | 2025-01-13 |
| 1.1.2 | 2025-04-03 |
| 1.1.3 | 2025-08-05 |
| 1.2.0 | 2026-09-04 ("Compatibility with transformers v5, sentence-transformers v5.4 and v6, huggingface_hub v1"; Python 3.13; ONNX/OpenVINO export fixes) |

- Sources: [PyPI release history](https://pypi.org/project/setfit/#history); [GitHub releases](https://github.com/huggingface/setfit/releases)
- Note: the WebFetch summarizer of the GitHub releases page mislabeled the years (it said 2024 for v1.2.0). I used the PyPI dates, which are authoritative and match the local git log.

**Maintainers and tests.** `setup.py` lists "Lewis Tunstall, Tom Aarsen". Every compatibility release since 2024 was authored by Tom Aarsen. The test suite has about 139 test functions. There are 12 notebooks: text classification, multilabel, hyperparameter search, ABSA FiQA, ONNX, Optimum, Optimum-Intel, OpenVINO and zero-shot. Source: local repo; [setup.py](https://github.com/huggingface/setfit/blob/main/setup.py)

### Inferences
- Since 2024, SetFit has been in "keep-the-lights-on" mode. The single active maintainer (Tom Aarsen, who also leads ST) has spent each release absorbing breaking changes in transformers, ST, hub and datasets. None of these releases added features.
- The architecture is already half absorbed into ST: phase 1 is ST's trainer. The SetFit-specific value that remains is:
  - pair generation (`ContrastiveDataset`)
  - the two-phase orchestration and tuple-args UX
  - the sklearn head and its persistence
  - ABSA
  - distillation
  - exporters
  - model cards
- The parts with the most technical debt:
  - The hand-written torch loop for the differentiable head (no multi-GPU, AMP, logging or checkpointing).
  - Pickle-based head serialization, a security and safetensors gap.
  - The eager O(n²) pair materialization, the cause of the OOM reports.
  - The ONNX export path, pinned to the legacy torch exporter at opset 12 and to hummingbird-ml.
  - The spaCy-dependent ABSA.
  - Dead ST < 2.3 branches.

### Gaps
- Commits before Feb 2024 are not visible locally because the clone is shallow. The exact date of the last big feature push before v1.0 (late 2023) is not confirmed from the git log.
- I did not run the test suite, so I have no runtime confirmation of the compatibility claims against ST 6.1.

## 2. Most-engaged open issues and PRs; recurring feature requests

### Takeaway
On GitHub (Oct 2026) the repo has about 145 open issues and 13 open PRs. Engagement per issue is low: the top open issue has about 7 reactions. The demand that does recur is for:
- multi-GPU or large-scale training
- regression and similarity/pair tasks
- custom pairs and losses (MNRL, GISTEmbed)
- instruction/prompt-aware embedding models
- memory and scale problems in pair generation
- ONNX and multi-label export correctness
- Hub and loading errors

A new multimodal-body PR (#653) appeared in Sep 2026.

### Cited Findings
**Repo stats.** 2.8k stars, 270 forks, 145 open issues, 13 open PRs, about 730 commits. Source: [GitHub repo page](https://github.com/huggingface/setfit), fetched 2026-10-01.

**Top open issues by reactions.**
1. #91 "Using Setfit for similarity classification" (2022-10-07, 7 reactions, 6 comments)
2. #541 "setfit model on multiple GPUs?" (2024-07-17, 6 reactions, 1 comment)
3. #94 "HFValidationError when loading the model" (20 comments)
4. #126 "Does num_iterations create duplicate data?"
5. #254 "Why are the models fine-tuned with CosineSimilarity between 0 and 1?" (10 comments)
6. #120 "Using SetFit Embeddings for Semantic Search?"
7. #167 "Add conda-forge package"
8. #520 "exception: enum PyPreTokenizerTypeWrapper, while loading the fine-tuned model"
9. #398 "hyperparameters to control how to handle long documents"
10. #264 "Trainer crashes on GCP"
11. #382 "Train SBERT using 2 sentences as input for detecting duplicates"
12. #560 "Opset Issue while exporting to onnx"
13. #291 "ONNX export for multilabel — discrepancy"

Other open issues in the top 25: #144 "[FR] use MultipleNegativesRankingLoss", #499 "Multioutput target data is not supported with label binarization", #462 "Inconsistency between predict and predict_proba with string labels", #526 "Model checkpoints saved during training are unusable", #567 "Trainer does not release all CUDA memory".
- Sources: [issues sorted by reactions](https://github.com/huggingface/setfit/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-desc); [GitHub search API via MCP](https://github.com/huggingface/setfit/issues/91)

**Top open issues by comments.**
1. #94 HFValidationError (20)
2. #315 "Choosing the datapoints that need to be annotated?" (19), i.e. active learning
3. #408 "No tutorial or guideline for Few-shot learning on multiclass text classification" (15)
4. #226 column-mapping ValueError (12)
5. #224 "model file marked as unsafe" on Hub (12), the pickle issue
6. #291 ONNX multilabel (11)
7. #192 "How to use a custom Sentence Transformer pretrained model" (11)
8. #498 `model.config.to_dict()` AttributeError (11)
9. #254 (10)
10. #335 "Prediction Interpretability" (9)
11. #316 "Multi-label topics not captured" (9)
12. #375 "How to weight loss function" (9), i.e. class imbalance
13. #425 "Error with model.predict() when using early stopping" (9)
14. #472 "Kernel crash due to out of memory for large dataset" (9)
15. #355 ONNX multi-output (8)
16. #567 CUDA memory not released (8)
17. #314 "train and deploy via Sagemaker" (7)
18. #119 "Using SetFit for regression tasks?" (6)

- Source: [issues sorted by comments](https://github.com/huggingface/setfit/issues?q=is%3Aissue+is%3Aopen+sort%3Acomments-desc)

**Other open feature requests.** All have low engagement:
- #604 "Instructor-style models (NV-Embed, e5-large-instruct, SFR-Embedding-2_R, etc.)" (2025-05-26)
- #575 "Use of SetFit within the browser" (2024-11-30)
- #576 "Specifying ones own samples or sentence pairs" (2024-12-16)
- #395 "setfit for ranking problems?"
- #328 "Shap Values with Setfit"
- #539 "setfit can not get good result for chinese language?"
- #209 "Limitations of Setfit Model"
- #616 "Setfit classifier trained on a static-embeddings model?" (Aug 2025, closed)
- #422 "Oversized required packages" (closed)

Source: [GitHub issue search via MCP, huggingface/setfit](https://github.com/huggingface/setfit/issues/604)

**Open PRs, in reactions order.**
- #152 "[WIP] add option to use MultipleNegativesRankingLoss" (PhilipMay, Nov 2022)
- #599 "Ensure consistency between labels and probabilities" (Apr 2025)
- #412 "Fixed switched token_type_ids and attention_mask" (Aug 2023, 5 comments)
- #661 zero-shot tutorial typo (2026-09-24)
- #653 "Task-aware encoding and image inputs for multimodal Sentence Transformer bodies" (oneryalcin, 2026-09-11)
- #627 "Fix OOM in contrastive pair generation with streaming approach" (2026-01-22)
- #642 "Apply TrainingArguments.max_length during embedding training" (2026-06-16)
- #646 "Exclude identity pairs from contrastive pair generation" (2026-08-27)
- #579 "Refactor ContrastiveDataset and ContrastiveDistillationDataset" (Dec 2024)
- #662 "Keep a one-example class from turning the contrastive loss into NaN" (2026-09-29)
- #584 "add option to use guided loss GISTEmbedLoss" (Jan 2025)
- Two bot PRs (#657, #658)

Source: [open PRs](https://github.com/huggingface/setfit/pulls?q=is%3Apr+is%3Aopen+sort%3Areactions-desc)

### Inferences
- The community-contributed PRs cluster on the sampler and loss layer: streaming pairs, identity pairs, NaN with singleton classes, MNRL and GIST losses. Some of these PRs are 1–4 years old and still unmerged. This points to a review-bandwidth bottleneck, not a lack of contributor interest.
- PR #627 (OOM) and PR #642 (max_length is ignored during embedding training) suggest real correctness and scale bugs in the current trainer path. If #642's title is accurate, `TrainingArguments.max_length` is not applied to phase 1.
- Requested capabilities with no implementation today:
  - multi-GPU
  - regression
  - pair/similarity classification
  - custom pairs
  - instruction-aware encoding (prompts)
  - multimodal bodies
  - browser/ONNX-first deployment
  - active learning
  - interpretability
  - class weighting
- Several of these already exist natively in ST v3–v6: prompts, multi-GPU training via the HF Trainer, MNRL and GIST losses, and multimodal support. That makes them cheap wins if SetFit exposes the ST machinery directly.
- Engagement is modest overall: no issue has more than about 7 reactions. Demand is broad and shallow, not concentrated on one feature.

### Gaps
- The GitHub search API (`/search/issues`) was blocked in this session, so exact reaction totals for every issue could not be pulled programmatically. Ordering comes from GitHub's web UI sort, and reaction counts are confirmed only for #91 (7) and #541 (6) via the MCP search.
- GitHub Discussions and Hub forum threads were not reviewed.

## 3. Adoption metrics

### Takeaway
SetFit is used widely but modestly compared with its parent library:
- about 2.8k GitHub stars
- about 146k PyPI downloads in the last 30 days, versus about 22.7M for sentence-transformers
- about 3k models on the Hub with `library:setfit`
- about 332 citations for the paper (Semantic Scholar)

Downloads were stable at 210–260k per month from Apr–Aug 2026, then fell sharply in Sep 2026.

### Cited Findings
**PyPI downloads (pypistats, 2026-10-01).**
- Recent: last_day 5,938; last_week 31,747; last_month 146,244. Source: [pypistats API, setfit recent](https://pypistats.org/api/packages/setfit/recent)
- Monthly, without mirrors:

| Month | Downloads |
|---|---|
| 2026-04 | 212,112 |
| 2026-05 | 225,706 |
| 2026-06 | 226,097 |
| 2026-07 | 261,153 |
| 2026-08 | 238,111 |
| 2026-09 | 139,725 |

  March 2026 is partial (19,081) because the window starts there. Source: [pypistats overall](https://pypistats.org/api/packages/setfit/overall?mirrors=false)
- Python-version split over the window: 3.10 (544,608), 3.11 (327,356), 3.12 (271,210), 3.13 (95,698), 3.14 (43,127), 3.9 (8,678). Source: [pypistats python_minor](https://pypistats.org/api/packages/setfit/python_minor)

**Comparisons.**
- sentence-transformers: last_month 22,690,816. Source: [pypistats sentence-transformers](https://pypistats.org/api/packages/sentence-transformers/recent)
- IBM `fastfit`: last_month 14. Source: [pypistats fastfit](https://pypistats.org/api/packages/fastfit/recent)

**Hugging Face Hub.**
- The UI shows "2,990" models for `library=setfit`. Source: [HF models?library=setfit](https://huggingface.co/models?library=setfit&sort=downloads)
- The most-downloaded tagged models have small numbers:
  - deutsche-telekom/gbert-large-paraphrase-cosine (17.4k; actually an ST model carrying the setfit tag)
  - SetFit/test-setfit-sst2 (5.62k; a test fixture)
  - tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect (5.01k) and -polarity (4.91k)
  - KBLab/emotional-classification (1.71k)
  - Source: [same](https://huggingface.co/models?library=setfit&sort=downloads)

**Paper and GitHub.**
- Semantic Scholar: citationCount 332, influentialCitationCount 51 for "Efficient Few-Shot Learning Without Prompts". Source: [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2209.11055?fields=title,citationCount,influentialCitationCount)
- GitHub: 2.8k stars, 270 forks. Source: [GitHub](https://github.com/huggingface/setfit)

### Inferences
- About 146k–260k monthly downloads alongside only about 3k Hub models suggests SetFit is mostly used privately or in CI and pipelines (download counts include CI installs), not as a public model-sharing ecosystem.
- Top Hub downloads are in the low thousands, and many of the top models are test fixtures or the maintainer's own demos. There is no widely shared flagship SetFit model.
- The September 2026 drop (about 140k, down from about 240k) coincides with the v1.2.0 release on 2026-09-04 and the ST v6 and transformers v5 floor changes. It could reflect pypistats data lag, a CI-driven artifact, or real attrition. It cannot be attributed with confidence from this data.
- Download share relative to ST is about 0.6%.

### Gaps
- Google Scholar citation count was not retrieved; only Semantic Scholar's 332 is available.
- pepy.tech total and all-time downloads, and pre-March-2026 trend data, were not retrieved because pypistats only keeps 180 days.
- GitHub "Used by" (dependents) count was not retrieved.
- Notable companies using SetFit in production: I found no reliable primary sources. A search for Argilla and enterprise case studies returned irrelevant results.
- Intel Labs co-authored the paper (Korat, Wasserblat, Pereg) and contributed the Optimum-Intel and OpenVINO notebooks; see the [arXiv](https://arxiv.org/abs/2209.11055) author list. That is the clearest institutional tie.

## 4. Relationship to Sentence Transformers' newer training stack (obsolescence risk)

### Takeaway
ST has grown far beyond SetFit's original base, and SetFit is now a thin orchestration layer over ST's trainer. Recent ST releases added a series of new capabilities:
- v3 SentenceTransformerTrainer
- v4 CrossEncoder training
- v5 SparseEncoder
- v5.4 multimodal and VLM support
- v5.5 an agent "train-sentence-transformers" skill and EmbedDistillLoss
- v5.7 GradCache rework and token-based mini-batching
- v6.0 MultiVectorEncoder/ColBERT, with transformers v5 and torch 2.2+ as the floor

SetFit uses almost none of these beyond the basic trainer and a few losses.

### Cited Findings
**ST release timeline (PyPI upload dates).**

| Version | Date |
|---|---|
| 5.2.3 | 2026-02-17 |
| 5.3.0 | 2026-03-12 |
| 5.4.0 | 2026-04-09 |
| 5.5.0 | 2026-05-12 |
| 5.6.0 | 2026-06-16 |
| 5.6.1 | 2026-07-23 |
| 5.7.0 | 2026-08-06 |
| 6.0.0 | 2026-08-18 |
| 6.0.1 | 2026-08-31 |
| 6.1.0 | 2026-09-18 (current) |

Source: [PyPI JSON sentence-transformers](https://pypi.org/pypi/sentence-transformers/json)

**ST v6.0.0.** Titled "MultiVectorEncoder for ColBERT & late interaction models, transformers v5, float32 scoring, faster training & encoding". It raises the floor to transformers v5 and torch 2.2+. Sources: [ST releases](https://github.com/huggingface/sentence-transformers/releases); [ai-tldr summary](https://ai-tldr.dev/releases/sentence-transformers-6-0/)

**ST v5.4.0.**
- "Full multimodal support for both SentenceTransformer and CrossEncoder".
- A `model.modalities` / `supports()` API.
- A `LogitScore` module for CausalLM rerankers.
- Flash-Attention-2 padding skipping.
- It moved modules to `sentence_transformers.sentence_transformer.*`, which SetFit had to shim in `compat.py`.
- Sources: [ST releases](https://github.com/huggingface/sentence-transformers/releases); [setfit compat.py](https://github.com/huggingface/setfit/blob/main/src/setfit/compat.py)

**ST v5.5–v5.7.**
- v5.5: `EmbedDistillLoss`, `ADRMSELoss`, and the "train-sentence-transformers" Agent Skill.
- v5.6: hard-negative mining and GIST loss fixes, MPS support for cached losses.
- v5.7: GradCache engine rebuild, `mini_batch_num_tokens` (a reported 3.9x speedup), and `model.compile()`.
- Source: [ST releases](https://github.com/huggingface/sentence-transformers/releases)

**What SetFit actually uses.**
- `SentenceTransformerTrainer`, `SentenceTransformerTrainingArguments`, `losses` (CosineSimilarity, batch-triplet family), `BatchSamplers.GROUP_BY_LABEL` and the `Dense` module.
- No prompts or instructions, no Matryoshka, no CrossEncoder, no SparseEncoder, no multimodal inputs and no Router or task-aware encoding. Open PR #653 tries to add task-aware and multimodal encoding.
- Sources: [compat.py](https://github.com/huggingface/setfit/blob/main/src/setfit/compat.py); [trainer.py](https://github.com/huggingface/setfit/blob/main/src/setfit/trainer.py); [PR list](https://github.com/huggingface/setfit/pulls)

### Inferences
- The risk of obsolescence is real. A user can already reproduce SetFit's phase 1 with plain ST: build pairs, then `SentenceTransformerTrainer` with `CosineSimilarityLoss` or `BatchHardTripletLoss` plus `GROUP_BY_LABEL`. Phase 2 is a few lines of sklearn.
- SetFit's distinctive value today is packaging: the pair sampler, two-phase UX, head persistence, model cards, ABSA, distillation and exporters. Its research novelty is not the draw.
- Conversely, ST's newer features map directly onto SetFit's open requests:
  - prompts → instruction models (#604)
  - multi-GPU via HF Trainer/accelerate → #541
  - MNRL/GIST losses → #144, #152, #584
  - multimodal → #653
  - CrossEncoder → pair/similarity classification (#91, #382)
  - Matryoshka → smaller and faster heads
- A transformational update could make SetFit "ST-native": train the head inside ST's trainer, save it as an ST module, and drop the pickle.
- Every major ST or transformers release has forced a SetFit compatibility release (v1.1.1, v1.1.2, v1.1.3, v1.2.0). The maintenance burden is roughly proportional to the size of SetFit's custom glue: the callback monkey-patching and the separate `TrainingArguments`.

### Gaps
- I did not confirm whether ST 6.x ships any built-in single-text classification head or SetFit-equivalent recipe. The release notes I reviewed show none, but ST docs were not exhaustively checked.
- Whether ST maintainers plan to fold SetFit into ST is unknown; no public statement was found.

## 5. Original paper and blog claims, and which are outdated in the LLM era

### Takeaway
The 2022 paper and blog benchmarked SetFit against 2021–22 few-shot baselines: GPT-3, T-Few 3B/11B, PET, ADAPET, PERFECT and fine-tuned RoBERTa. The bodies were MPNet (110M) and RoBERTa-Large (355M) STs.

Its headline claims ("outperforms GPT-3", "27x smaller than T-Few", "$0.025 to train") were made before instruction-tuned LLMs (GPT-4-class, Llama 3+, Qwen) and before modern embedding models (BGE, E5, GTE, NV-Embed, ModernBERT). The baselines are therefore stale. The efficiency argument still holds, and arguably matters more now.

### Cited Findings
**Abstract.** SetFit "works by first fine-tuning a pretrained ST on a small number of text pairs, in a contrastive Siamese manner… then used to generate rich text embeddings, which are used to train a classification head… obtains comparable results with PEFT and PET techniques, while being an order of magnitude faster to train."
- Authors: Tunstall, Reimers, Jo, Bates, Korat, Wasserblat, Pereg.
- Submitted 2022-09-22; there is only v1 on arXiv.
- Source: [arXiv 2209.11055](https://arxiv.org/abs/2209.11055)

**RAFT leaderboard snapshot (Sep 2022), from the blog.**

| Method | Score | Size |
|---|---|---|
| T-Few | 75.8% | 11B |
| Human baseline | 73.5% | — |
| SetFit (RoBERTa-Large) | 71.3% | 355M |
| PET | 69.6% | 235M |
| SetFit (MPNet) | 66.9% | 110M |
| GPT-3 | 62.7% | 175B |

Source: [HF blog "SetFit: Efficient Few-Shot Learning Without Prompts"](https://huggingface.co/blog/setfit)

**Other blog claims.**
- "comparable results to T-Few 3B, despite being prompt-free and 27 times smaller".
- With 8 examples per class on Customer Reviews, it is "competitive with fine-tuning RoBERTa Large on the full training set of 3k examples".
- Training costs: SetFit on a V100 takes 30 s and about $0.025, versus T-Few 3B on an A100 at 11 min and about $0.70.
- "Distilling the SetFit model can bring speed-ups of 123x."
- Source: [HF blog](https://huggingface.co/blog/setfit)

**Current README.** It still leads with comparisons to "T0 or GPT-3" and with "No prompts or verbalizers". Source: [README](https://github.com/huggingface/setfit/blob/main/README.md)

**Benchmark artifacts in the repo.** The repo still contains `final_results/` for adapet, tfew, perfect, setfit, distillation and transformers, plus a `scripts/` directory of paper-era experiment scripts. Source: local repo listing ([final_results](https://github.com/huggingface/setfit/tree/main/final_results))

### Inferences
- Outdated claims:
  - "Outperforms GPT-3". Zero- and few-shot GPT-4-class and open 7–70B instruction-tuned models are very likely much stronger on RAFT-style tasks than GPT-3 davinci was.
  - The "No prompts" framing. Modern embedding models are instruction-prompted, and prompts are now a strength, not a liability.
  - Default bodies like `paraphrase-mpnet-base-v2`. These are now far behind BGE, GTE, E5, Nomic, ModernBERT-based and static-embedding models.
- Durable claims:
  - Training cost of seconds to minutes on one GPU or CPU.
  - Inference cost, latency, privacy and on-prem operation versus LLM APIs.
  - Determinism and small model size, which suit edge, browser and ONNX deployment.
- A refreshed benchmark against LLM zero/few-shot baselines and modern embedding bodies would be needed to restate the value proposition.

### Gaps
- I found no recent (2024–2026) peer-reviewed head-to-head benchmark of SetFit versus GPT-4-class or open LLMs. Search results surfaced only the 2022 GPT-3 comparison and secondary articles ([TDS](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e), [philschmid](https://www.philschmid.de/getting-started-setfit)).
- The full paper tables (per-dataset results, multilingual MARC results) were not extracted.

## 6. Forks, successors, and community extensions

### Takeaway
The best-known direct successor or competitor is IBM's FastFit (2024). It claims 3–20x faster training than SetFit on many-class problems, but has minimal adoption: 222 stars and about 14 PyPI downloads per month. No dominant SetFit fork or successor was found. Most "successor" pressure comes from LLM prompting and from plain ST fine-tuning.

### Cited Findings
**FastFit (IBM, arXiv 2404.12365, "When LLMs are Unfit Use FastFit").**
- It integrates "batch contrastive learning and token-level similarity score".
- It encodes texts and class names into a shared space.
- It claims a "3-20x improvement in training speed" over SetFit, Transformers and LLM few-shot prompting.
- Sources: [arXiv 2404.12365](https://arxiv.org/pdf/2404.12365); [IBM Research](https://research.ibm.com/publications/fastfit-fast-and-effective-few-shot-text-classification-with-a-multitude-of-classes)
- FastFit's GitHub has 222 stars and 35 commits (fetched 2026-10-01). Source: [IBM/fastfit](https://github.com/IBM/fastfit)
- PyPI `fastfit` had 14 downloads in the last month. Source: [pypistats fastfit](https://pypistats.org/api/packages/fastfit/recent)

**Third-party tutorials and wrappers.** Examples include a W&B report, width.ai, anote.ai few-shot docs, and the tessl registry documentation of `pypi-setfit`. These show SetFit referenced as a standard few-shot recipe in third-party tooling. Sources: [W&B report](https://wandb.ai/gladiator/SetFit/reports/SetFit-Efficient-Few-Shot-Learning-Without-Prompts--VmlldzozMDUyMzk2); [width.ai](https://width.ai/post/what-is-setfit); [anote docs](https://docs.anote.ai/fewshot/fewshotclassification.html); [tessl registry](https://tessl.io/registry/tessl/pypi-setfit)

**Intel ecosystem.** The repo notebooks cover Optimum-Intel, OpenVINO and ONNX/Optimum, contributed by Intel Labs co-authors (#484, #517, #496). Source: [notebooks](https://github.com/huggingface/setfit/tree/main/notebooks)

### Inferences
- The competitive landscape is less "another SetFit-like library" and more "LLM prompting for zero-shot" plus "ST fine-tuning for embeddings".
- FastFit's idea of label-name embeddings with token-level similarity is a technique SetFit could absorb, for example as a label-aware or many-class mode.

### Gaps
- I could not verify an Argilla–SetFit integration in this session; the search returned irrelevant results. Argilla historically published SetFit tutorials, but no source was retrieved here.
- I did not survey GitHub forks of huggingface/setfit (270) for notable divergent work.
- Other community extensions were not found: SetFit-for-NER forks, SetFit in spaCy-llm, browser/transformers.js ports (#575 asks for one), or Rust ports.
