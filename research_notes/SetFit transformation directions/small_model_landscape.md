# Small / Efficient Text Classification Landscape (as of Oct 2026) — competitors and backbones for SetFit

Scope: encoder, embedding and other non-generative approaches only. LLM-based classification and distillation are covered elsewhere. Dates are given for every finding. Where a fetch tool returned dates that conflict with other sources, this is noted.

## 1. Modern encoder backbones: which suit few-shot classification fine-tuning?

### Takeaway
Since late 2024 the encoder layer has turned over. ModernBERT, then Ettin (Jul 2025), mmBERT (Sep 2025) and EmbeddingGemma / Qwen3-Embedding (mid/late 2025) all have 8K-class context and permissive licenses (except Gemma). Few-shot evidence favours *embedding-tuned* checkpoints such as modernbert-embed-base, EmbeddingGemma and Qwen3-Embedding-0.6B as SetFit bodies. *Raw MLM* encoders (Ettin, mmBERT, ModernBERT) remain the better choice for full fine-tuning or as bases for training new sentence encoders. One caution from Knowledgator: DeBERTa-v3 still beat ModernBERT as a zero-shot classifier backbone in their tests.

### Cited Findings
- **Ettin (JHU/LightOn, 16 Jul 2025):** paired encoder-only and decoder-only models from 17M to 1B parameters (17M/32M/68M/150M/400M/1B). All have 8K context, Apache-2.0 license and fully open training data, trained on 2T tokens in three phases (1.7T + 250B + 100B) — [HF blog: Ettin](https://huggingface.co/blog/ettin)
- Ettin's authors say its encoders "outperform ModernBERT across all tasks and model sizes". On MNLI a 150M Ettin encoder scores 89.2, ahead of a 400M Ettin decoder at 88.2. Converting an encoder to a decoder or the reverse ("cross-training") underperforms native models — [HF blog: Ettin](https://huggingface.co/blog/ettin); [HF papers 2507.11412](https://huggingface.co/papers/2507.11412); [LightOn blog](https://www.lighton.ai/lighton-blogs/introducing-ettin-suite-the-sota-open-recipe-to-outperform-existing-generative-retrieval-models)
- **mmBERT (JHU, Sep 2025):** a ModernBERT-architecture encoder pretrained on 3T tokens covering more than 1,800 languages. It is described as "the first to significantly improve upon XLM-R", scores 80.54 on XNLI, and is MIT-licensed. Over 1,700 low-resource languages were added only during the decay phase — [arXiv 2509.06888](https://arxiv.org/abs/2509.06888); [InfoQ, Sep 2025](https://www.infoq.com/news/2025/09/mmbert); [HF: jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small)
- A newer multilingual follow-on, **MrBERT** (arXiv Feb 2026), adapts modern multilingual encoders through vocabulary, domain and dimensional changes. Its results were not checked — [arXiv 2602.21379](https://arxiv.org/pdf/2602.21379)
- **EmbeddingGemma (Google, Sep 2025, paper arXiv 2509.20354):** reports "exceptional improvements in classification (+8.5)" over other models on MTEB(Eng, v2) and claims first place among sub-1B models for classification. The license is the Gemma license, not OSI-approved — [arXiv 2509.20354](https://arxiv.org/pdf/2509.20354); [model card license: gemma](https://huggingface.co/lazyDataScientist/EmbeddingGemma/blob/main/README.md)
- **Qwen3-Embedding (Jun 2025):** MTEB(eng, v2) classification scores are 0.6B → 85.76, 4B → 89.84 and 8B → 90.43 — [arXiv 2506.05176](https://arxiv.org/pdf/2506.05176). A secondary catalog gives a different figure (66.83) for 0.6B, probably measured on a different MTEB variant (multilingual) — [Accenture model catalog](https://sdk.airefinery.accenture.com/distiller/model_catalog/Embedding/Qwen/Qwen3-Embedding-0.6B/)
- **SetFit + ModernBERT (community benchmark, Intel's Moshe Wasserblat, ~early 2025):** SetFit on `modernbert-embed-base` beat plain ModernBERT fine-tuning by a wide margin in few-shot settings. On IMDB with 8 samples per class it scored 92.7% against 62.5%; on ArXiv-new with 64 samples per class it scored 90.3% against 76.5% — [Medium (search snippet; full page returned 403)](https://moshewasserblat.medium.com/new-results-on-setfit-modernbert-for-text-classification-with-few-shot-training-53c154df7c0e)
- Knowledgator built GLiClass on both DeBERTa-v3 and ModernBERT and reports that "DeBERTa-based models consistently outperform those based on ModernBERT" for zero-shot classification, despite ModernBERT's Flash Attention and longer context (Aug 2025) — [GLiClass paper, arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)

### Inferences
- SetFit's best default bodies are now embedding-tuned models built on modern encoders, not raw MLM checkpoints. The paper's IMDB result shows the contrastive/embedding stage is what produces the few-shot gain. Candidate defaults:
  - modernbert-embed / Ettin-based embedders and EmbeddingGemma-300M for English;
  - Qwen3-Embedding-0.6B for quality-first, multilingual use;
  - mmBERT-derived embedders for multilingual use with an MIT license.
- The Ettin 17M–68M encoders are natural "tiny" backbones for SetFit. Nothing comparable existed before (MiniLM was the only small option).
- License matters to SetFit's enterprise users. Ettin (Apache-2.0) and mmBERT (MIT) are clean; EmbeddingGemma's Gemma license is a friction point.

### Gaps
- No public benchmark was found that runs SetFit across Ettin, mmBERT, EmbeddingGemma, Qwen3-Embedding, NeoBERT, EuroBERT and Arctic-Embed on RAFT or few-shot splits. This is an open gap and an opportunity for SetFit to publish one.
- Detailed specs for NeoBERT, EuroBERT, Jina v3/v4, Nomic, GTE, BGE and Arctic-Embed were not retrieved in this pass, and the current MTEB classification leaderboard ranking (Oct 2026) was not fetched directly.

## 2. Zero-shot / few-shot classifier families (GLiClass, GLiNER2, NLI zero-shot, FastFit)

### Takeaway
SetFit's most direct competitors are "label-aware" encoders that need zero or few examples:
- **GLiClass** (Knowledgator): GLiNER-style, single forward pass over all labels.
- **GLiNER2** (Fastino): schema-driven classification plus extraction.
- **NLI universal classifiers** (MoritzLaurer zeroshot-v2.0).
- **FastFit** (IBM): many-class few-shot.

GLiClass and FastFit both publish head-to-head numbers against SetFit. SetFit still has the fastest *inference*, but it is weak with zero labeled examples and with many similar classes.

### Cited Findings
- **GLiClass v3.0 (paper 11 Aug 2025):**
  - **Architecture:** a uni-encoder that prepends a `<<LABEL>>` token to each label, concatenates the labels with the text, and runs one bidirectional pass. This gives label-label and text-label interaction and multi-label output in a single forward pass. Pooling and scoring use either a dot-product or an MLP scorer.
  - **Variants explored:** bi-encoder, fused bi-encoder and encoder-decoder.
  - **Training:** a 1.2M-example pretraining corpus, then RL (PPO) mid-training, then LoRA post-training on logic/NLI and on synthetic label-density data annotated by GPT-4o.
  - — [arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)
- **GLiClass v3.0 model sizes, average F1 on 14 benchmarks, and throughput (A6000, batch size 1):**

  | Model | Params | Avg F1 | Throughput |
  |---|---|---|---|
  | edge | 32.7M | 0.4900 | 97.29 ex/s |
  | modern-base | 151M | 0.5577 | 54.46 ex/s |
  | modern-large | 399M | 0.6197 | 43.80 ex/s |
  | base | 187M | 0.6764 | 51.61 ex/s |
  | large | 439M | 0.7193 | 25.22 ex/s |

  gliclass-large-v3.0 beats deberta-v3-large-zeroshot-v2.0 (0.6821) by +0.037 absolute — [arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)
- **GLiClass few-shot (8 examples per label):** gains over zero-shot were +0.19 for edge (+50%), +0.21 for modern-base, +0.19 for modern-large, +0.11 for base and +0.11 for large — [arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)
- **GLiClass label scaling:** going from 1 to 128 labels, gliclass-edge throughput fell 103.8 → 82.6 ex/s (-20%) and base fell 49.4 → 45.9 ex/s (-7%). Over the same range the cross-encoder deberta-v3-base-zeroshot-v2.0 fell from 24.55 to 0.47 ex/s, about 52 times slower. Overall GLiClass is 2.3–16 times faster than cross-encoders — [arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)
- GLiClass model cards include comparisons with SetFit on IMDB, AG_NEWS and Emotions. The larger GLiClass models generally beat SetFit there. The cards also say "SetFit will be faster depending on model size and amount of labels" — [HF: gliclass-modern-large-v3.0](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0); [GitHub Knowledgator/GLiClass](https://github.com/Knowledgator/GLiClass). Exact numbers were not extracted.
- A community Space compares "GliZNet vs GLiClass", which suggests derivative architectures are appearing — [HF Space](https://alexneakameni-zero-shot-classification.hf.space/)
- **GLiNER2 (Fastino, arXiv Jul 2025, EMNLP 2025 demo):** a single 205M-parameter model that unifies NER, text classification, structured extraction and relation extraction behind a schema API (label descriptions, thresholds, choices). It runs on CPU and installs with pip. On zero-shot classification across SST-2, IMDB, SNIPS, Banking77, Amazon-Intent, AG News and 20NG it "achiev[es] the highest average accuracy among open-source baselines" — [arXiv 2507.18546](https://arxiv.org/pdf/2507.18546); [HF fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1); [ACL Anthology EMNLP demos](https://preview.aclanthology.org/ingest-emnlp/2025.emnlp-demos.10.pdf)
- A "fastino/GLiNER2.5-Decide" model (intent classification plus extraction, ~34.7K downloads) appeared in an auto-generated HF-trending digest on 1 Oct 2026. It is unverified on the Hub — [agents-radar digest 2026-10-01](https://github.com/THTHDGCS/agents-radar/issues/1060)
- **NLI "universal classifiers" (MoritzLaurer, paper Dec 2023, arXiv 2312.17543):**
  - deberta-v3-zeroshot-v2.0 is trained on 33 datasets with 389 classes, mixing synthetic data from Mixtral with commercially friendly NLI data. It improves zero-shot performance by 9.4% and was evaluated on 28 tasks (f1_macro) against facebook/bart-large-mnli.
  - The "-c" variants are commercially licensed; there is also a multilingual bge-m3-zeroshot-v2.0.
  - The older zero-shot classifiers had more than 55M Hub downloads as of Dec 2023.
  - — [Paper summary](https://huggingface-gw-kz.just-ai.com/papers/2312.17543.md); [HF bge-m3-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c)
- **FastFit (IBM + Hebrew University, NAACL 2024 demo, arXiv 2404.12365):**
  - Method: batch contrastive learning plus a token-level similarity score, aimed at many semantically similar classes.
  - **Speed:** training is 3–20 times faster than SetFit and finishes in seconds.
  - **Accuracy (5-shot):** FastFit-large beats SetFit by 2% and a standard classifier by 4.5%.
  - **Accuracy (10-shot):** it beats SetFit by 2.7% and a standard classifier by 0.7%. FastFit-small matches SetFit-large at 5-shot and beats it at 10-shot.
  - **Benchmark:** FewMany (English) plus multilingual sets.
  - — [arXiv 2404.12365](https://arxiv.org/pdf/2404.12365); [IBM Research](https://research.ibm.com/publications/fastfit-fast-and-effective-few-shot-text-classification-with-a-multitude-of-classes)
- **Other comparisons:** a 2024 Polish few-shot benchmark compares fine-tuning, linear probing, SetFit and in-context learning — [arXiv 2404.17832](https://arxiv.org/abs/2404.17832). An Oct 2024 study covers efficient few-shot multi-label classification of scientific documents with many classes — [arXiv 2410.05770](https://arxiv.org/html/2410.05770v1)

### Inferences
- GLiClass and GLiNER2 hit SetFit where it is weakest: zero labeled examples, label sets that change at inference time, multi-label tasks with label interactions, and label descriptions. SetFit's edge is that it beats them after a few examples, and its inference is a plain bi-encoder plus head with no label-count overhead.
- One distinctive direction for SetFit is a hybrid: initialize SetFit's head from label-description embeddings so it works at zero-shot, then improve with few-shot contrastive tuning. This would cover the whole 0→N-shot curve that GLiClass covers, and keep bi-encoder inference speed.
- FastFit's token-level similarity and batch-contrastive loss directly answer SetFit's known weakness on many-class (50+) tasks. It is a candidate loss or head to adopt.

### Gaps
- No up-to-date (2026) download or star counts were obtained for GLiClass, FastFit or GLiNER2. GitHub API access was blocked in this session.
- No independent (non-vendor) benchmark was found comparing GLiClass with SetFit at matched shots.
- "tasksource" models were not researched in this pass.

## 3. Static embeddings & distillation for speed (model2vec, StaticEmbedding, WordLlama)

### Takeaway
Static embeddings are now the "fast" end of the market. model2vec's full-finetune classifier trains and infers about 35 times faster than SetFit on CPU, giving up only about 3.4 points of accuracy. sentence-transformers' StaticEmbedding gives ~400 times CPU speedups with ~86% of classification quality retained. SetFit has no static-embedding path, which leaves this segment to MinishLab.

### Cited Findings
- **model2vec classification results** (14 datasets, 1,000 training examples each, no tuning):

  | Model | Avg accuracy | Throughput (samples/s, CPU) |
  |---|---|---|
  | TF-IDF | 70.8% | 108,434 |
  | Model2Vec + LogReg | 78.0% | 17,925 |
  | Model2Vec full finetune (StaticModelForClassification) | 79.2% | 24,744 |
  | SetFit | 82.6% | 716 |
  | BGE-base + LogReg | 82.8% | 118 |

  The finetuned model2vec is described as 35 times faster than SetFit (all-MiniLM-L6-v2) on CPU — [minish.ai results](https://minish.ai/packages/model2vec/results/)
- On MTEB classification, potion-base-32M scores 71.70 against 69.25 for all-MiniLM-L6-v2 — [minish.ai results](https://minish.ai/packages/model2vec/results/)
- StaticModelForClassification fine-tunes the static embedding weights as well as the head; the vanilla path is LogisticRegressionCV on frozen embeddings — [model2vec PR #186](https://github.com/MinishLab/model2vec/pull/186/files)
- MinishLab positions model2vec as a fastText alternative (blog dated 28 Jul 2025) — [minish.ai blog](https://minish.ai/blog/2025-07-28-fasttext/)
- A third-party distillation guide (sieves.ai) estimates that SetFit students keep 80–95% of teacher accuracy with training in minutes, while model2vec students keep 70–85% with training in seconds. This is vendor guidance, not a controlled benchmark — [sieves.ai](https://sieves.ai/guides/distillation/)
- **sentence-transformers StaticEmbedding (15 Jan 2025):**
  - Released models: static-retrieval-mrl-en-v1 and static-similarity-mrl-multilingual-v1.
  - **Speed:** 397 times faster on CPU and 24 times faster on GPU than all-mpnet-base-v2; about 125 times faster on CPU than multilingual-e5-small.
  - **Quality retained vs multilingual-e5-small:** classification 86.52%, pair classification 95.52%, STS 92.3%.
  - Can be initialized `from_model2vec`, but the released models were trained from scratch.
  - — [HF blog: static embeddings](https://huggingface.co/blog/static-embeddings)

### Inferences
- SetFit could add a "static student" mode that distills a trained SetFit model into a model2vec / StaticEmbedding body. This would provide both quality (SetFit teacher) and fastText-level speed (static student), and neither MinishLab nor sentence-transformers packages the combination today.
- Accuracy gap: SetFit leads model2vec-finetune by about 3.4 points at 1,000 examples. That margin probably grows at 8–64 shots, where SetFit is strongest, but no published numbers confirm it (see Gaps).

### Gaps
- No few-shot (8/16/64-shot) comparison of model2vec against SetFit was found; model2vec's benchmark uses 1,000 examples.
- WordLlama was not researched in this pass.
- Current model2vec star and download counts were not obtained because GitHub API access was blocked.

## 4. sentence-transformers v3 → v6.1: does it overlap with SetFit?

### Takeaway
sentence-transformers (now under the huggingface org) is a full training framework for four model types:
- SentenceTransformer;
- CrossEncoder (v4);
- SparseEncoder (v5);
- MultiVectorEncoder (ColBERT-style, v6.0, Aug 2026).

It also supports multimodal input, ONNX/OpenVINO backends, static embeddings and agent skills. No native few-shot classification head or SetFit-style "contrastive pairs → head" pipeline was found in any release. SetFit's value proposition is thinning but not yet absorbed.

### Cited Findings
- **v6.1.0** was released 18 Sep 2026, with refreshed inference benchmarks and better multimodal input handling — [newreleases v5.6/6.x](https://newreleases.io/project/github/huggingface/sentence-transformers/release/v5.6.0); [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases)
- **v6.0.0 (Aug 2026):**
  - New `MultiVectorEncoder` for ColBERT-style late interaction (one vector per token, MaxSim) across text, image, audio and video. `lightonai/LateOn` scores NanoBEIR NDCG@10 0.6868 against 0.6764 for its dense counterpart.
  - Now requires transformers>=5.0 and torch>=2.2.
  - **Speed:** training is about 1.22–1.26 times faster, and fp16 + Flash Attention GPU inference is 3.87 times faster than fp32.
  - **Correctness:** fixed float32 scoring for half-precision embeddings.
  - The release notes do not mention SetFit.
  - v6.0.1 widened PyLate checkpoint loading (51 → 80 documented multi-vector models).
  - — [newreleases v6.0.0](https://newreleases.io/project/github/huggingface/sentence-transformers/release/v6.0.0); [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases); [ai-tldr summary](https://ai-tldr.dev/releases/sentence-transformers-6-0/)
- **v5.7.0:** GradCache overhaul (up to 3.9 times faster cached-loss training) and `model.compile()` for inference — [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases)
- **v5.6.0 (Jul 2026):** correctness fixes (causal-LM reranker scoring, hard-negative mining, GIST loss), TSDAE restored on transformers v5, and MPS support for cached losses — [newreleases v5.6.0](https://newreleases.io/project/github/huggingface/sentence-transformers/release/v5.6.0)
- **v5.5.0:** a `train-sentence-transformers` Agent Skill for AI coding agents, `EmbedDistillLoss` for embedding-level distillation, and `ADRMSELoss` for listwise CrossEncoder training — [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases)
- **v5.4.0 (Apr 2026):**
  - First-class multimodal support (text, image, audio, video).
  - A fully modular CrossEncoder with a `LogitScore` module for generative rerankers.
  - Flash Attention 2 skips padding.
  - ONNX/OpenVINO backends.
  - — [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases); [mindwiredai, 13 Apr 2026](https://mindwiredai.com/2026/04/13/sentence-transformers-just-went-multimodal-heres-why-its-a-big-deal/)
- Note on dates: the GitHub-releases fetch tool labelled these releases "2024". The search index and the v6 dependency on transformers v5 (a 2026-era floor) point to 2026. The 2026 dates are treated as correct here.
- Router module and prompts allow asymmetric query/document paths and task-specific routes; the StaticEmbedding and SparseEncoder modules are also present — [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases)

### Inferences
- Overlap: sentence-transformers already covers SetFit's stage 1 (contrastive fine-tuning with modern losses, prompts, Matryoshka, static and ONNX backends), and does it better and faster. SetFit's remaining unique pieces are:
  - the few-shot pair-generation recipe;
  - the integrated differentiable/sklearn head;
  - the one-call `SetFitModel.predict` classification UX;
  - few-shot-specific tooling (ABSA, multi-label strategies, knowledge distillation).
- Risk: one "SentenceTransformerForClassification" or "classification head module" in sentence-transformers would largely absorb SetFit. Opportunity: SetFit can lean into sentence-transformers' new primitives:
  - SparseEncoder bodies for interpretable classifiers;
  - MultiVectorEncoder bodies for token-level / FastFit-like scoring;
  - multimodal bodies for image+text few-shot classification;
  - the new Router for multi-task setups.
- The fact that sentence-transformers ships an "Agent Skill" shows the maintainers see AI coding agents as the new users. SetFit could do the same.

### Gaps
- No public sentence-transformers roadmap or issue was found that proposes a native classification head. Not verified.
- Exact release dates for v3.0 (May 2024), v4 (CrossEncoder training, ~Mar 2025) and v5 (SparseEncoder, ~mid 2025) were not re-fetched in this pass; they are approximate from background knowledge.

## 5. Late interaction / rerankers (ColBERT, PyLate) for classification

### Takeaway
Late-interaction models are now mainstream infrastructure: PyLate checkpoints load natively in sentence-transformers v6. Their use for classification is mostly *indirect*. Cross-encoder rerankers double as zero-shot classifiers (NLI style), and GLiClass explicitly markets itself as a reranker. FastFit's token-level similarity is the closest published "late interaction for classification" method.

### Cited Findings
- sentence-transformers v6.0 ships MultiVectorEncoder (MaxSim) and loads PyLate checkpoints; 80 multi-vector models are documented as of v6.0.1 — [newreleases v6.0.0](https://newreleases.io/project/github/huggingface/sentence-transformers/release/v6.0.0); [GitHub releases](https://github.com/UKPLab/sentence-transformers/releases)
- GLiClass is described as usable "for topic classification, sentiment analysis, and as a reranker in RAG pipelines" — [HF: gliclass-modern-large-v3.0](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0)
- FastFit uses a token-level similarity score (ColBERT-like) with batch contrastive learning for many-class few-shot classification — [arXiv 2404.12365](https://arxiv.org/pdf/2404.12365)
- Cross-encoder zero-shot classification scales badly with label count: deberta-v3-base-zeroshot-v2.0 falls from 24.55 to 0.47 ex/s going from 1 to 128 labels — [arXiv 2508.07662](https://www.arxiv.org/pdf/2508.07662)

### Inferences
- A "MultiVector SetFit" is a plausible gap: few-shot contrastive training of a MaxSim body against label prototypes (or label descriptions). It would combine FastFit's accuracy on many classes with sentence-transformers' new infrastructure.

### Gaps
- No 2025–2026 paper was found that uses ColBERT/PyLate models directly as few-shot text classifiers.

## 6. Benchmarks 2024–2026 comparing SetFit with these approaches

### Takeaway
Head-to-head evidence is scattered and mostly vendor-authored:
- GLiClass (zero/8-shot, Aug 2025);
- FastFit (5/10-shot, 2024);
- model2vec (1,000-shot, ~2025);
- one community ModernBERT test (~early 2025).

No neutral 2025–2026 benchmark covering all of these at matched shots was found. This is an opening for SetFit to publish a definitive "few-shot classification leaderboard".

### Cited Findings
- **FastFit vs SetFit:** +2.0% (5-shot) and +2.7% (10-shot) for FastFit-large; training 3–20 times faster — [arXiv 2404.12365](https://arxiv.org/pdf/2404.12365)
- **model2vec vs SetFit:** 79.2% vs 82.6% average over 14 datasets at 1,000 examples; model2vec 35 times faster — [minish.ai](https://minish.ai/packages/model2vec/results/)
- **GLiClass vs SetFit:** GLiClass large models score higher on IMDB, AG_NEWS and Emotions per the model cards; SetFit is faster — [HF gliclass model card](https://huggingface.co/knowledgator/gliclass-modern-large-v3.0)
- **SetFit (modernbert-embed-base) vs plain ModernBERT fine-tuning:** 92.7% vs 62.5% on IMDB with 8 samples per class — [Medium](https://moshewasserblat.medium.com/new-results-on-setfit-modernbert-for-text-classification-with-few-shot-training-53c154df7c0e)
- **Polish few-shot benchmark (2024):** compares SetFit, linear probing, fine-tuning and ICL — [arXiv 2404.17832](https://arxiv.org/abs/2404.17832)
- **Original SetFit paper (2022):** the baseline reference; RAFT results — [arXiv 2209.11055](https://arxiv.org/pdf/2209.11055)

### Inferences
- Each competitor picks the regime where it wins:
  - GLiClass: zero-shot;
  - FastFit: many classes;
  - model2vec: throughput at 1,000+ examples;
  - SetFit: 8–64 shots with a modern body.
- A neutral, reproducible benchmark spanning 0/8/64/1,000 shots, accuracy against CPU latency, and English plus multilingual data would be high-leverage for SetFit.

### Gaps
- No 2025–2026 RAFT leaderboard update involving these methods was found.
- No MTEB-style "few-shot classification" track was found.

## 7. Trending / viral small-model classification projects (2025–2026)

### Takeaway
The verifiable momentum in 2025–2026 sits in four places:
- **GLiNER-family** "one small model, schema/labels at inference" tools (GLiNER2, GLiClass, GLiNER2.5-derived variants);
- **static-embedding speed plays** (model2vec);
- **open modern encoders** (Ettin, mmBERT);
- **sentence-transformers' expansion** (multimodal, multi-vector).

What drove popularity: zero-shot usability via natural-language labels or schemas, CPU-only speed, pip-install simplicity, and permissive licenses. The HF-trending digests for late Sep 2026 list text-classification models with large like counts, but the data looks unreliable.

### Cited Findings
- Auto-generated HF-trending digests from late Sep / 1 Oct 2026 (agents-radar) list several text-classification models:
  - convaiinnovations/laya: ~4.1–4.7K likes but 0 downloads, "calibrated decision-making… System One framework";
  - laya-multilingual: ~306 likes;
  - AlexWortega/openjev: ~610 likes, described as an NLI-based cross-encoder for zero-shot classification;
  - akhilaaa3/Jev-Omni: Gemma 4-based;
  - SupersonicLabs/Julia-1;
  - fastino/GLiNER2.5-Decide: ~34.7K downloads.

  The 0-download / 4K-like profile and LLM-written descriptions make these unreliable — [agents-radar 2026-10-01](https://github.com/THTHDGCS/agents-radar/issues/1060); [agents-radar 2026-09-26](https://github.com/stevenko2002/agents-radar/issues/1457)
- GLiNER2 became an EMNLP 2025 system demo and ships as a pip library with a schema API — [ACL Anthology](https://preview.aclanthology.org/ingest-emnlp/2025.emnlp-demos.10.pdf); [Pioneer/Fastino blog](https://pioneer.ai/blog/gliner2)
- MinishLab markets model2vec as a fastText replacement (Jul 2025) — [minish.ai blog](https://minish.ai/blog/2025-07-28-fasttext/)
- sentence-transformers' multimodal release (v5.4, Apr 2026) drew mainstream tech-blog coverage — [mindwiredai](https://mindwiredai.com/2026/04/13/sentence-transformers-just-went-multimodal-heres-why-its-a-big-deal/)

### Inferences
- The popular pattern is "describe labels in natural language at inference time, run on CPU". SetFit requires training, so adding a zero-shot / label-description mode would align it with what is trending.
- The decision-model / "System One" framing (small calibrated classifiers acting as fast gates in agent pipelines) appears to be an emerging 2026 theme. If real, calibration and abstention would be differentiating SetFit features.

### Gaps
- GitHub star counts and growth (star-history) for setfit, model2vec, GLiClass, GLiNER2, FastFit, PyLate and WordLlama could not be retrieved: the GitHub API was blocked in this session (HTTP 403) and no star-history page was fetched. This is the biggest data gap for "traction".
- The HF-trending digest entries (laya, openjev, Julia-1, GLiNER2.5-Decide) were not verified against the Hub.
