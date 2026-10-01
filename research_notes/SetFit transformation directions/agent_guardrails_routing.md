# High-demand applications for small, fast, custom text classifiers in the 2025–2026 LLM/agent stack (as of 2026-10-01)

Method note: web search + primary-source fetches (model cards, arXiv, vendor docs) on 2026-10-01. Hugging Face Hub numbers come from the public HF API, queried 2026-10-01. The `downloads` field is a rolling 30-day count. GitHub API and code search were **blocked in this session**, so there are no GitHub star counts or code-search results (see Gaps).

## 1. Guardrails & safety classifiers: sizes, latencies, customization, and demand for few-shot custom-policy classifiers

### Takeaway
The guardrail market has split into two tiers. (a) Tiny encoder classifiers (22M–184M params, about 20–90 ms on GPU) handle fixed tasks like prompt injection and jailbreaks, and they carry by far the most download volume. (b) Generative "bring-your-own-policy" judges (0.6B–120B params, hundreds of ms to seconds) handle custom policies. Nothing in the middle is cheap, CPU-fast, and customizable from a handful of examples. Vendors concede that the policy-prompted LLM approach is slower and less accurate than a trained classifier, and that is exactly the gap a SetFit-style tool could fill.

### Cited Findings
**Small encoder guardrails (fixed-policy)**
- Llama Prompt Guard 2 (released 2025-04): 86M model at 92.4 ms/classification on an A100 at 512 tokens, with 99.8% AUC (EN) and 97.5% recall @1% FPR. The 22M model (DeBERTa-xsmall) runs at 19.3 ms, cuts latency/compute by about 75%, and scores 99.5% AUC with 88.7% recall @1% FPR (EN) — [Groq model docs](https://console.groq.com/docs/model/llama-prompt-guard-2-22m); [HF mirror model card](https://huggingface.co/project-free-llama/Llama-Prompt-Guard-2-86M)
- HF API (2026-10-01), 30-day downloads: meta-llama/Prompt-Guard-86M (v1) 3,542,580; protectai/deberta-v3-base-prompt-injection-v2 (184M) 771,865; unitary/unbiased-toxic-roberta 953,843; Llama-Prompt-Guard-2-86M 125,278; Llama-Prompt-Guard-2-22M 16,594; ibm-granite/granite-guardian-hap-38m 1,192 — [HF API](https://huggingface.co/api/models?sort=downloads&direction=-1&limit=1000)
- Common practice in guardrail frameworks is to wrap an off-the-shelf HF text-classification model such as `ProtectAI/deberta-v3-base-prompt-injection-v2`, for example in W&B's guardrails-genie `PromptInjectionClassifierGuardrail` — [HF Space source](https://huggingface.co/spaces/wandb/guardrails-genie/blob/8f135f2c3f0b26be72501c4b10b84f4e3b7222bb/guardrails_genie/guardrails/injection/classifier_guardrail.py)
- Lakera Guard (commercial API) documents latency of <20 ms p95 for 1,000 characters, <100 ms for 10k, <150 ms for 20k and <200 ms for 30k characters. Network/TLS adds a reported 5–15 ms p50 and 30–60 ms p95 under load. — [Lakera latency benchmark docs](https://docs.lakera.ai/docs/latency-benchmark); network-overhead figure from [beri.net comparison (2026)](https://www.beri.net/article/nemo-guardrails-vs-guardrails-ai-vs-lakera-runtime-prompt-injection-2026), a secondary source

**Generative / LLM-based safety classifiers (fixed taxonomy)**
- Parameter counts from the HF API (safetensors total), 2026-10-01: Llama-Guard-4-12B 12.0B (46,058 dl/30d); Llama-Guard-3-8B 8.03B (46,063); Llama-Guard-3-1B 1.50B (25,372); google/shieldgemma-2b 2.61B (12,608); ibm-granite/granite-guardian-3.3-8b 8.17B (4,232); nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 8.03B (1,874) — [HF API model endpoints](https://huggingface.co/api/models/meta-llama/Llama-Guard-4-12B)
- Llama Guard 4 has a 164k context; ShieldGemma 9B has 8k — [llmreference comparison](https://www.llmreference.com/compare/llama-guard-4-12b/shieldgemma-9b)
- Llama Guard 3-1B-INT4 reaches a time-to-first-token of ≤2.5 s on a commodity Android mobile CPU. Even the smallest LLM guard is about 100× slower than encoder classifiers on CPU. — [arXiv 2411.17713](https://arxiv.org/pdf/2411.17713)
- Qwen3Guard (released 2025-09) comes in 0.6B, 4B and 8B sizes and was trained on 1.19M labeled prompts and responses. It has two variants: Gen (generative, instruction-following) and Stream (a token-level classification head for real-time moderation during generation). Labels are safe/controversial/unsafe across 119 languages. Stream requires Qwen3-tokenizer token IDs. — [Qwen3Guard-Stream-0.6B model card](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B); [arXiv 2510.14276](https://arxiv.org/pdf/2510.14276v1)
- Qwen3Guard-Gen-0.6B has 168,304 downloads/30d, the most-downloaded LLM-based guard in this sample. That suggests demand for small sizes. — [HF API](https://huggingface.co/api/models/Qwen/Qwen3Guard-Gen-0.6B)

**"Bring-your-own-policy" (BYOP) classifiers**
- gpt-oss-safeguard (120b and 20b, Apache 2.0). The policy is supplied at inference time and the model reasons over it. The 20b has 21B total / 3.6B active params, needs ≥16 GB VRAM, and offers low/medium/high reasoning effort. HF lists created 2025-09-18; the model card says released 2025-08-08 (inconsistent). It is part of the ROOST model community. — [HF model card](https://huggingface.co/openai/gpt-oss-safeguard-20b); [LM Studio](https://lmstudio.ai/blog/gpt-oss-safeguard)
- OpenAI's own stated limitations: "classifiers trained on tens of thousands of high-quality labeled samples can still perform better" than gpt-oss-safeguard reasoning from a policy. Also, "reasoning takes more compute and time, which makes it harder to use at large scale or in real-time." — [OpenAI technical report PDF](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf); [Help Net Security](https://www.helpnetsecurity.com/?p=346721). The openai.com page returned 403, so this was verified via secondary coverage and search snippets of the report.
- Pitch from secondary coverage: "rapidly calibrate safety rules just by editing the policy text, without having to train a whole new classifier (which costs a lot of time and money)" — [HackerNoon](https://hackernoon.com/openai-puts-guardrails-on-its-open-weight-gpts)
- NVIDIA Nemotron-Content-Safety-Reasoning-4B is a 4.3B model based on Gemma-3-4B-it. It is explicitly "bring your own safety policy", with "Reasoning Off" (low latency) and "Reasoning On" modes. — [HF model card](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B)
- NVIDIA Llama-3.1-NemoGuard-8B-TopicControl is a LoRA on Llama-3.1-8B-Instruct trained on synthetic topic-following data (Mixtral-8x7B). It takes a system prompt of allowed/disallowed topics and returns on-topic/off-topic, and is designed for "task-oriented dialogue agents and custom policy-based moderation". — [NeMo Guardrails docs](https://docs.nvidia.com/nemo/guardrails/latest/get-started/tutorials/nemoguard-topiccontrol-deployment); [NIM reference](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemoguard-8b-topic-control)
- Granite Guardian 3.0 is 8B with an 8k context — [llmreference](https://www.llmreference.com/compare/granite-guardian-3.0-8b/llama-guard-2-8b)

**Direct evidence of demand for few-shot custom guardrails**
- kNNGuard (arXiv 2607.02072, 2026-07-06) is a training-free guardrail built from a bank of **50 safe + 50 unsafe** examples. It uses multi-layer kNN over Llama-3.1-8B activations plus embeddings. Across six domains (code instructions, code outputs, medical, safety, jailbreak, prompt injection) it averages 87.4% F1 (e.g., 99.1% code instr., 95.3% medical, 73.7% safety, 75.0% jailbreak, 82.2% injection). Latency: kNNGuard 45.9 ms vs NemoGuard TopicControl 126.0 ms vs Nemotron Safety Guard V2 454.6 ms vs a plain embedding-kNN at **4.0 ms**. The bank is built in <10 s. Its motivation: "adapting to a new domain requires collecting a new training corpus, fine-tuning the model from scratch, and re-validating the classifier." — [alphaXiv 2607.02072](https://www.alphaxiv.org/abs/2607.02072.md); [HF papers](https://huggingface.co/papers/2607.02072)
- JurEE (arXiv 2410.08442) is an ensemble of small encoder-only classifiers, one per risk category, for a banking chatbot. Categories are Safe, Harmful, Off Topic, System Attack, Complaint and Vulnerable. It reached about 0.92 F1 vs 0.54–0.64 for baselines, including LLM judges. — [arXiv 2410.08442](https://arxiv.org/pdf/2410.08442)
- A 2026 paper titled "Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges" exists. The title was found; contents were not verified. — [arXiv 2606.25782](https://arxiv.org/pdf/2606.25782)
- Thoughtworks Technology Radar (Oct 2024) placed SetFit in **Trial**: "We've successfully used SetFit for intent detection in a customer-facing chatbot system that uses an LLM". They also used it "to perform additional fine-tuning to achieve stricter filtering" in place of commercial moderation APIs. — [Thoughtworks Radar](https://www.thoughtworks.com/radar/languages-and-frameworks/setfit)

### Inferences
- Demand for customizable policies is real and growing. Llama Guard, Qwen3Guard, NemoGuard TopicControl, Nemotron CS Reasoning 4B and gpt-oss-safeguard all moved toward taxonomy or policy customization in 2024–2025. But every customizable offering is a ≥0.6B generative model needing a GPU. The fixed-task models people actually deploy at scale are ≤184M encoders, judging by HF downloads where Prompt-Guard-86M v1 alone has 3.5M/30d.
- The open niche is "custom policy, encoder-speed, CPU-deployable, trained from 8–100 examples". That is SetFit's natural slot. kNNGuard's 50+50 example setup and its comparison against an embedding-kNN baseline are effectively a SetFit-shaped benchmark that SetFit was not included in.
- One plausible winning workflow: draft a policy, label with gpt-oss-safeguard or an LLM, then distil into a SetFit or small encoder. The FineWeb-Edu pattern (LLM labels → small classifier, see §5) is the precedent, and OpenAI's own limitation note supports it: trained classifiers win once labels exist.

### Gaps
- Could not fetch openai.com directly (403). No published latency numbers found for gpt-oss-safeguard, Llama Guard 4, Granite Guardian or NemoGuard TopicControl.
- No current data found on OpenAI omni-moderation latency/pricing, Guardrails AI Hub validator usage, or Protect AI/Lakera customer counts.
- No market-size figures (revenue, number of deployments) for guardrail products were found in reliable sources.

## 2. LLM routing: classifier-based vs embedding-based approaches, and training-data needs

### Takeaway
Routers range from embedding-similarity over hand-written example utterances (Aurelio semantic-router), through fine-tuned BERT/ModernBERT classifiers (RouteLLM, vLLM Semantic Router), to 1.5B generative routers (Arch-Router). Commercial routers (Not Diamond) train on as few as 25 labeled prompts. Routing therefore looks like a few-shot text-classification problem with a 50 ms budget, which is SetFit's core competency. Small-scale SetFit routers already exist on the Hub.

### Cited Findings
- RouteLLM (LMSYS/Berkeley/Anyscale, ICLR 2025) offers four routers: similarity-weighted ranking, matrix factorization, a BERT classifier (BERT-base embedding + logistic head, fine-tuned on preference data), and a causal-LLM classifier. Matrix factorization cut cost by >85% on MT-Bench at 95% of GPT-4 quality (14% of queries sent to the strong model). Cost cuts were 45% on MMLU and 35% on GSM8K. Training data was Chatbot Arena preference data plus augmentation. — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- routellm/bert_gpt4_augmented (278M) has 38,896 downloads/30d — [HF API](https://huggingface.co/api/models/routellm/bert_gpt4_augmented)
- vLLM Semantic Router (announced 2025-09) is an intent-aware routing layer that uses a ModernBERT classifier to decide routes, for example whether reasoning mode is needed. It reports ~10% higher accuracy, ~50% lower latency and ~50% fewer tokens, with >20% accuracy gains in business/economics. Classifier training uses MMLU-Pro (~12K samples, ~14 domains), Presidio PII (~50K token-level) and jailbreak datasets. — [vLLM blog](https://vllm.ai/blog/semantic-router); [arXiv 2510.08731](https://arxiv.org/pdf/2510.08731)
- Follow-up "98× Faster LLM Routing Without a Dedicated GPU" (arXiv 2603.12646, 2026): routing latency fell from 4,918 ms to 127 ms (flash attention), then 62 ms (prompt compression), then 50 ms (near-streaming), with 108 ms at 16K tokens on AMD MI300X. Its three router signals are safety classification, domain routing and PII detection. A search snippet said an mmBERT-32K variant runs as three concurrent ONNX sessions; the abstract fetch did not confirm this. — [arXiv 2603.12646](https://arxiv.org/pdf/2603.12646)
- Arch-Router (Katanemo, arXiv 2506.16655) is a 1.5B generative router. It maps queries to user-defined (domain, action) policies, so routes change without retraining. It claims to beat the best proprietary models on multi-turn routing benchmarks, with 50 ms p50 and <75 ms p99. HF: 1.54B params, 278 likes, 6,976 dl/30d. — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1); [HF](https://huggingface.co/katanemo/arch-router-1.5b)
- Aurelio semantic-router routes by embedding similarity. Each Route is a list of example "utterances" with a score_threshold, and the incoming query is matched to the nearest route. It cuts "route making time down from seconds to milliseconds." — [Aurelio docs](https://docs.aurelio.ai/semantic-router/user-guide/concepts/architecture); [The New Stack](https://thenewstack.io/semantic-router-and-its-role-in-designing-agentic-workflows/)
- Not Diamond custom routers need prompts, candidate-model responses and per-response evaluation scores, with a **minimum of 25 samples**. — [Not Diamond docs](https://docs.notdiamond.ai/docs/router-training-quickstart)
- SetFit-tagged routers on the HF Hub (2026-10-01) are all low-download: MajidFQ/taxoSplitter-biz-router-setfit-v1 (35 dl/30d), snival/intent-router-zh-setfit-v2 (21), cnmoro/prompt-router (19), egis-group/router_mini_lm_l6 (13), yaniseuranova/setfit-rag-hybrid-search-query-router (11), mrzaizai2k/model_routing_few_shot (12), boredpanda9/medical-query-router (8), naranor/SetFit-Multilingual-ONNX-Router-V1 — [HF API filter=setfit search=router](https://huggingface.co/api/models?filter=setfit&search=router&limit=50)

### Inferences
- Aurelio's "utterances per route" is functionally SetFit's input format (a few examples per class) without the contrastive fine-tuning step. SetFit could position itself as the trained upgrade to semantic-router, with the same data and better decision boundaries.
- RouteLLM's BERT router and vLLM-SR's ModernBERT router are full fine-tunes on 10K+ examples. For enterprise-specific routes (custom domains or tools) such data rarely exists, and that favors few-shot methods.
- Arch-Router's selling point is "no retraining when routes change". A SetFit router would need retraining, but in seconds to minutes on CPU. That is a competitive story only if training is near-instant and automated.

### Gaps
- No primary data found on Martian, OpenRouter Auto Router internals, or NotDiamond's classifier architecture.
- No published head-to-head of SetFit vs semantic-router vs a fine-tuned BERT router on routing benchmarks.
- GitHub stars for RouteLLM, semantic-router, archgw and vllm-project/semantic-router could not be retrieved (API blocked).

## 3. Agent-system classifiers: intent/tool selection, topic control, RAG query classification, hallucination/groundedness, PII, jailbreak, small judges, observability tagging

### Takeaway
Every agent pipeline now has several "is this X?" decision points. Specialized small models exist for groundedness (HHEM, LettuceDetect), small judges (Glider 3.8B, Luna-2) and PII/jailbreak (vLLM-SR heads). For the long tail of app-specific labels, such as intents, off-topic, needs-retrieval and custom eval tags, practitioners still mostly use LLM calls. Recent evidence shows SetFit beating a 20B LLM on intent classification at about 1,700× lower latency.

### Cited Findings
**Intent / agent caching / topic**
- "Why Agent Caching Fails…" (arXiv 2602.18922, Feb–Mar 2026) uses **SetFit with 8 examples per class** for intent canonicalization in agent caching. It reports 91.1% ±1.7% accuracy on MASSIVE at ~2 ms, vs a 20B LLM at 68.8% and 3,447 ms, and GPTCache at 37.9%. On NyayaBench v2 (20 classes, cross-lingual) it scores 55.3%. — [arXiv 2602.18922](https://arxiv.org/abs/2602.18922)
- TrueFoundry "Enterprise-Grade Intent Classification with SetFit" (Jan 2026) recommends 5–10 examples per class. It targets sub-100 ms p95 with 350+ RPS on 1 vCPU, and cites 86.1% accuracy on a legal domain. Use cases are customer-inquiry routing, fraud, adverse-event flagging and ticket prioritization. — [TrueFoundry blog](https://www.truefoundry.com/blog/truefoundry-accelerator-series-building-enterprise-grade-intent-classification-with-setfit)
- NemoGuard TopicControl (8B LoRA) is NVIDIA's off-topic detector (see §1). kNNGuard measured it at 126 ms/prompt. — [alphaXiv 2607.02072](https://www.alphaxiv.org/abs/2607.02072.md)

**Hallucination / groundedness**
- LettuceDetect (KRLabs, arXiv 2502.17125, Feb 2025) is a ModernBERT token classifier with 8,192-token context, trained on RAGTruth. It reaches 79.22% example-level F1, which is +14.8% over Luna. It processes 30–60 examples/s on one GPU and is about 30× smaller than the best LLM-based detectors. HF base model is 150M params with 35,593 dl/30d. — [arXiv 2502.17125](https://arxiv.org/html/2502.17125v1); [HF API](https://huggingface.co/api/models/KRLabsOrg/lettucedect-base-modernbert-en-v1)
- Vectara HHEM-2.1-Open (FLAN-T5-based, 110M) needs <600 MB RAM and takes about 1.5 s for 2k-token inputs on a standard CPU. It has 148,164 dl/30d. — [Vectara blog](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/); [HF API](https://huggingface.co/api/models/vectara/hallucination_evaluation_model)

**Small judges / eval**
- Galileo Luna-2 runs fine-tuned small decoder SLMs with LoRA heads per metric. It scores in <200 ms vs ~2,600 ms for GPT-4-based eval, at $0.02 vs $0.15 per M tokens (about 97% cheaper, per Galileo's claim). Metrics include toxicity, hallucination and tool-selection quality. Enterprise tier only. — [Galileo Luna docs](https://docs.galileo.ai/concepts/luna/luna.md); [arXiv 2602.18583](https://www.arxiv.org/pdf/2602.18583)
- Patronus GLIDER (Dec 2024) is a 3.8B evaluator for arbitrary user-defined criteria, with higher Pearson correlation than GPT-4o on FLASK and comparable to LLMs 17× its size. HF: 810 dl/30d. Lynx-8B (hallucination judge) has 327 dl/30d. — [GLIDER paper listing](https://www.catalyzex.com/paper/glider-grading-llm-interactions-and-decisions); [HF API](https://huggingface.co/api/models/PatronusAI/glider)
- A "tiny-judge" design replaces a 32B judge with 8 task-specific LoRA adapters on ModernBERT-large, about 10× fewer judge parameters. The source is a search snippet; the paper was not individually verified. — via search; also see [arXiv 2403.02839](https://arxiv.org/pdf/2403.02839), which finds fine-tuned judges are "not a general substitute for GPT-4" (a counterpoint)

**PII / jailbreak in-router**
- vLLM Semantic Router runs PII (token-level, Presidio-derived ~50K examples) and jailbreak classifiers alongside domain classification in the request path — [vLLM blog](https://vllm.ai/blog/semantic-router)

**SetFit models for agent tasks on the HF Hub (2026-10-01, 30-day downloads)**
- spidercob/dlp-intent-classifier 336; spidercob/code-risk-classifier 418; waterabbit114/my-setfit-classifier_toxic 1,946; alirezaaminzadeh/agentshield-{prompt-injection 23, policy-decision 21, tool-risk 19, content-trust 18}; NaveenSandaruwanJayasooriya/tool-poisoning-detection 17 and wso2/tool-poisoning-detection 12 (MCP tool-poisoning); Pranab11/ollive-guardrail-ensemble-{prompt_injection, role_override, out_of_scope, sensitive_bias} (3–9 each); Myadav/setfit-prompt-injection-MiniLM-L3-v2 22; Revesis/setfit-mpnet-nli-rag-hallucination 10; ilyankou/is-geospatial-query 12; onyx-dot-app/information-content-model 1,237 (used in the Onyx enterprise-search product to score chunk information content) — [HF API filter=setfit](https://huggingface.co/api/models?filter=setfit&limit=1000&sort=downloads&direction=-1)

### Inferences
- The "agentshield" and "ollive-guardrail" repos are multi-head SetFit ensembles for injection, role-override, out-of-scope, tool-risk and policy-decision. Together with wso2's tool-poisoning model, they show organic, grassroots use of SetFit for agent guardrails, including MCP and tool security. They are tiny in adoption, though.
- Groundedness and judge tasks need cross-encoding of (context, claim) pairs over long inputs, and LettuceDetect and HHEM handle that. SetFit's sentence-embedding bi-encoder design is a weaker fit there unless SetFit adds pair or cross-encoder inputs and long-context backbones (ModernBERT/mmBERT).
- The best-fitting agent tasks for SetFit are single-text, many-custom-label decisions: intent, tool/skill pre-selection, off-topic, needs-retrieval, user-frustration/escalation, and observability tags.

### Gaps
- No sources found on how Langfuse, Arize or LangSmith implement automatic content tagging (LLM-based vs classifier-based).
- No benchmark found for "needs retrieval?" classifiers specifically (e.g., Self-RAG/Adaptive-RAG classifier sizes) in this research pass.
- No adoption numbers found for GLiNER/Presidio PII models.

## 4. Latency/cost budgets: why guardrail/routing classifiers must run in ~10–50 ms (CPU), and how often teams fine-tune

### Takeaway
Practitioner guidance in 2025–2026 converges on a 50 ms total guardrail budget, or <100 ms p50 for input checks. Classifiers run at 20–100 ms, LLM-based checks at 200 ms–10 s. A stack of several checks can double p95 latency and add about 40% to the inference bill. That pushes teams toward tiered cascades with small classifiers in front.

### Cited Findings
- "The Guardrail Tax" (2026-07-04) recommends a ~50 ms budget for the guardrail subsystem. A tiered setup adds <30 ms on the common path, while one example input classifier at 200 ms delays TTFT. A four-classifier stack doubled p95 latency and raised the inference bill by 40%, and lightweight classifiers should cost under a couple of percent of primary-model spend. Accuracy tiers: regex 60–70%, fine-tuned classifiers 89–94%, LLM judges slightly higher. Recommended patterns are concurrent execution, tiering (deterministic → small classifier → LLM judge), chunked streaming moderation and speculative generation. — [tianpan.co](https://tianpan.co/blog/2026/07/04/the-guardrail-tax)
- Production guardrail targets: sub-100 ms p50 for input and sub-150 ms p50 for output. A real-time chatbot needs <200 ms total guardrail overhead. Rule-based checks take µs–10 ms, classifiers 20–100 ms, LLM-based checks 500 ms–10 s, and LLM-powered guardrails are 5–10× slower than classifier-based ones. — [Future AGI guide 2026](https://futureagi.com/blog/ultimate-guide-llm-guardrails-2026/); [PremAI comparison](https://blog.premai.io/production-llm-guardrails-nemo-guardrails-ai-llama-guard-compared); [jacar.es](https://jacar.es/en/llm-guardrails-frameworks-and-their-real-cost/) (secondary/vendor blogs)
- A validator calling an auxiliary LLM adds 200–800 ms per turn, vs tens of ms for a small local classifier — same sources as above
- MiniGuard-v0.1 is reported to reach 99%+ of Nemotron-8B guard performance while being 13× smaller, 2–4× faster and about 3× cheaper — via search snippet of [Future AGI guide](https://futureagi.com/blog/ultimate-guide-llm-guardrails-2026/) (not independently verified)
- Measured latencies in this report: SetFit intent ~2 ms vs 20B LLM 3,447 ms ([arXiv 2602.18922](https://arxiv.org/abs/2602.18922)); embedding-kNN 4.0 ms, kNNGuard 45.9 ms, NemoGuard Topic 126 ms, Nemotron Safety Guard 454.6 ms ([kNNGuard](https://www.alphaxiv.org/abs/2607.02072.md)); Prompt Guard 2 22M 19.3 ms, 86M 92.4 ms on A100 ([Groq docs](https://console.groq.com/docs/model/llama-prompt-guard-2-22m)); vLLM-SR routing 50 ms after optimization, from 4,918 ms ([arXiv 2603.12646](https://arxiv.org/pdf/2603.12646)); Arch-Router 50 ms p50 ([arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)); Luna-2 <200 ms ([Galileo](https://docs.galileo.ai/concepts/luna/luna.md)); HHEM ~1.5 s for 2k tokens on CPU ([Vectara](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/)); TrueFoundry SetFit 350+ RPS on 1 vCPU ([TrueFoundry](https://www.truefoundry.com/blog/truefoundry-accelerator-series-building-enterprise-grade-intent-classification-with-setfit))

### Inferences
- Only encoder or embedding classifiers (≤~150M params) reliably fit a ≤50 ms budget on CPU. Anything ≥0.6B effectively needs a GPU. This is SetFit's structural advantage: MiniLM/bge-small backbones at 2–20 ms on CPU, plus ONNX/OpenVINO export.
- Several checks run per turn (injection + topic + PII + routing + output safety), so per-check latency multiplies. Multi-head or multi-label models that share one encoder pass are valuable, as vLLM-SR's three heads on a shared stack illustrate.

### Gaps
- No survey data found on what fraction of teams fine-tune their own guardrail classifiers vs use off-the-shelf or API ones. The question "how often teams fine-tune" is unanswered by reliable sources.

## 5. Data-pipeline classifiers at scale (FineWeb-Edu, NeMo Curator, DCLM fastText)

### Takeaway
Pretraining data curation runs classifiers over billions to trillions of tokens, so throughput dominates. The standard recipe is: label ~500K docs with a large LLM, then train a frozen-embedding + linear head (FineWeb-Edu) or a fastText model (DCLM). This is architecturally almost identical to SetFit's head-on-embeddings design. SetFit's few-shot contrastive step matters less here, because LLM labels are cheap and plentiful.

### Cited Findings
- FineWeb-Edu classifier: Llama-3-70B-Instruct scored about 460K FineWeb pages (CC-MAIN-2024-10) for educational value on a 0–5 scale. A linear regression head was trained on Snowflake-arctic-embed-m with frozen embedding/encoder layers, on 410K annotations for 20 epochs at lr 3e-4. A threshold of 3 kept 1.3T tokens and removed ~92% of the data, at ~82% F1 for the binary ≥3 cut. — [FineWeb paper arXiv 2406.17557](https://arxiv.org/pdf/2406.17557); HF: HuggingFaceFW/fineweb-edu-classifier, 109M params, 14,742 dl/30d, 228 likes ([HF API](https://huggingface.co/api/models/HuggingFaceFW/fineweb-edu-classifier))
- NeMo Curator Quality Classifier is DeBERTa-v3-base (184M) with a 1024-token context and High/Medium/Low labels from human annotators. NeMo Curator also ships a domain classifier and FineWeb-Nemotron-Edu classifiers. nvidia/quality-classifier-deberta has 4,712 dl/30d and nvidia/domain-classifier 8,393. — [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/quality-classifier-deberta); [NeMo Curator docs](https://docs.nvidia.com/nemo/curator/curate-text/process-data/quality-assessment/distributed-classifier); [HF API](https://huggingface.co/api/models/nvidia/quality-classifier-deberta)
- DCLM-Baseline uses a fastText classifier (mlfoundations/fasttext-oh-eli5) that separates "hq" (OpenHermes-2.5 + Reddit ELI5) from Common Crawl. It outperformed more sophisticated filters such as AskLLM and is much cheaper at inference. — [DCLM arXiv 2406.11794](https://arxiv.org/pdf/2406.11794); [HF model](https://huggingface.co/mlfoundations/fasttext-oh-eli5)
- Marin community published a "fastText vs BERT" quality-filter comparison report — [W&B report](https://wandb.ai/marin-community/marin/reports/Fasttext-vs-BERT--VmlldzoxMjgyOTk0OQ) (contents not fetched)
- Few-shot SetFit for data filtering already appears on the Hub: davanstrien/finepdfs-edu-purpose-classifier (60 dl/30d), mbley/german-webtext-quality-classifier-small, docmparker/all-mpnet-base-v2-setfit-8label-edu — [HF API filter=setfit](https://huggingface.co/api/models?filter=setfit&limit=1000&sort=downloads&direction=-1)

### Inferences
- SetFit's niche in data pipelines is *new, niche filters*: domain-specific SFT selection, a new language, a specific document "purpose". In these cases there is no LLM-labeled corpus yet, and 20–100 human examples plus fast iteration beat paying for 500K LLM annotations. davanstrien's finepdfs classifier is an example of this pattern from an HF team member.
- At web scale, throughput favors fastText or frozen-embedding + linear head. SetFit inference (a sentence-transformer encoder + head) matches the FineWeb-Edu design, so it is not disqualified on cost. Batch GPU inference and ONNX export would be required.

### Gaps
- No throughput numbers (docs/sec per GPU) were collected for the FineWeb-Edu or NeMo Curator classifiers.
- No published comparison found of few-shot (SetFit) vs LLM-annotated large-sample classifiers for pretraining filtering.

## 6. Evidence of practitioners currently using SetFit for these tasks

### Takeaway
SetFit usage for guardrails and routing is real but grassroots and small. There are dozens of Hub models (injection, toxicity, routing, intent, tool-poisoning, DLP), an arXiv paper (agent caching), a Thoughtworks Radar "Trial" endorsement, and enterprise vendor blogs. But aggregate adoption is orders of magnitude below single off-the-shelf guardrail models. SetFit is not currently the default tool in any of these categories.

### Cited Findings
- HF Hub (2026-10-01): the top 1,000 models tagged `setfit` total only **~75,288 downloads per 30 days** combined. The top SetFit model (deutsche-telekom/gbert-large-paraphrase-cosine) has 17,357. By comparison, Prompt-Guard-86M has 3,542,580 and protectai/deberta-v3-base-prompt-injection-v2 has 771,865 over the same window. — [HF API filter=setfit](https://huggingface.co/api/models?filter=setfit&limit=1000&sort=downloads&direction=-1)
- Hub search hits among SetFit-tagged models: "guardrail" 5, "prompt-injection" 5, "jailbreak" 0, "router" 13, "intent" 40 — [HF API](https://huggingface.co/api/models?filter=setfit&search=intent&limit=50)
- The most-downloaded guardrail-like SetFit models are waterabbit114/my-setfit-classifier_toxic (1,946/30d), spidercob/code-risk-classifier (418) and spidercob/dlp-intent-classifier (336). Everything else is <100. — same HF API source
- arXiv 2602.18922 (2026) uses SetFit with 8 examples per class for agent intent canonicalization: 91.1% on MASSIVE at ~2 ms vs a 20B LLM at 68.8% and 3,447 ms — [arXiv](https://arxiv.org/abs/2602.18922)
- Thoughtworks Radar Oct 2024: SetFit in "Trial", used for LLM-chatbot intent detection and stricter content filtering — [Thoughtworks](https://www.thoughtworks.com/radar/languages-and-frameworks/setfit)
- TrueFoundry (Jan 2026) sells a SetFit intent-classification accelerator — [TrueFoundry](https://www.truefoundry.com/blog/truefoundry-accelerator-series-building-enterprise-grade-intent-classification-with-setfit)
- None of the major guardrail frameworks or routers examined (NeMo Guardrails, Llama Guard/Prompt Guard, Qwen3Guard, gpt-oss-safeguard, RouteLLM, vLLM-SR, Arch-Router, semantic-router, Not Diamond) were found to use SetFit. kNNGuard (2026) benchmarks a generic embedding-kNN few-shot baseline but not SetFit. — [kNNGuard](https://www.alphaxiv.org/abs/2607.02072.md)

### Inferences
- The positioning gap: research such as kNNGuard and production vendors such as gpt-oss-safeguard and Nemotron CS Reasoning are converging on "custom policy from few examples or a text policy". SetFit is absent from these comparisons and integrations. Concrete moves could change that: integrations (NeMo Guardrails action, Guardrails AI validator, semantic-router encoder upgrade, vLLM-SR plugin), ONNX/CPU-first packaging, multi-head shared-encoder models, and LLM-label distillation (policy → synthetic labels → SetFit). Those would make SetFit the default "custom guardrail/router in minutes" tool.
- Biggest sized opportunities, ranked by deployment volume and fit:
  1. Prompt-injection and custom-topic guardrails. Volume: millions of downloads/month for fixed models. SetFit fit: custom domains.
  2. Intent, tool and route classification in agents. Fit: high; best evidence: arXiv 2602.18922.
  3. Niche data-curation filters.
  4. Lightweight eval/observability tagging.
  Groundedness/hallucination is a weaker fit without cross-encoder and long-context support.

### Gaps
- GitHub code search ("from setfit import" + guardrail/router) and star counts were blocked in this session (HTTP 403 from the session's GitHub proxy). A follow-up should run them from an unrestricted environment.
- HF download counts undercount private and enterprise use, and local/ONNX deployments don't register. No survey data found on SetFit's enterprise adoption.
