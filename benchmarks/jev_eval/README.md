# SetFit vs Jev and small LLMs on jev-eval

Compares few-shot and zero-shot SetFit with TypeSafe's Jev (`jev-1.13.0`), `gpt-5.4-mini` and `gpt-5.6-luna` on the
exact test items of [onlyoneaman/jev-eval](https://github.com/onlyoneaman/jev-eval): 300 items each from Enron spam,
SST-2, AG News and Banking77. That repo commits every case and every model's answer, so SetFit is scored on the same
items without calling any API.

- **Test items**: the committed cases (`--n_test N` takes the first N, which the probe sets to 100).
- **Training items**: sampled per class from each dataset's *train* split. The test items come from the test split (validation for SST-2), and exact-text overlaps are removed anyway.
- **Zero-shot**: SetFit's templated synthetic examples, built from the same one-line label descriptions Jev and the LLMs received.
- **Inputs**: identical to jev-eval's, including `Subject: …\n\n<message[:4000]>` for Enron.

## Run

```bash
cd benchmarks/jev_eval
./fetch_jev_eval.sh                       # pinned jev-eval checkout (cases + Jev/LLM answers)
uv venv && source .venv/bin/activate
uv pip install torch                      # Linux CPU-only: --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ../.. "optimum-intel[openvino]" openvino onnxruntime scikit-learn

# probe: 100 test items per dataset, 8 examples/class
./run.sh --model BAAI/bge-base-en-v1.5 --datasets sst2,ag_news,enron_spam,banking77 --shots 8 \
  --n_test 100 --batch_size 16 --max_steps 600 --latency 50 --out out_probe
python summarize.py out_probe
```

`--device auto` picks CUDA, then Apple MPS, then CPU. `--precision auto` uses bf16 where the hardware has it natively,
fp16 on CUDA GPUs without bf16, and fp32 otherwise, including MPS until bf16 is benchmarked there
(`--precision bf16` to try it). Latency is measured at batch size 1 for PyTorch, OpenVINO and OpenVINO int8, in a child
process: OpenVINO's exporter crashes under tcmalloc, which `run.sh` preloads on Linux.

## Caveats

- **Banking77 covers only 25 of 77 intents.** jev-eval samples whole 100-row pages and the test split is grouped by intent. Every model was still offered all 77 options, and SetFit trains on all 77.
- **Enron is truncated.** `--max_length 256` (the default) cuts long emails, while Jev saw up to 4,000 characters.
- **SetFit's probabilities are less extreme than Jev's.** With a logistic-regression head, none of SetFit's answers reached 0.9 confidence, so "accurate when confident" needs a calibrated threshold before it compares fairly.
- **Memory.** On a 16 GB machine, Enron at batch size 32 and 256 tokens peaked at about 13.7 GB with bge-small. Use batch size 16.

## Results so far (CPU probe, 100 items per dataset, seed 0)

Recorded on a 4-core Xeon (AVX-512 + VNNI, no bf16/AMX) with 16 GB RAM, fp32.

- `probe_bge-small_cpu` used batch size 32.
- `probe_bge-base_cpu` used batch size 16. Its training times are inflated where another job shared the CPU.
- The bge-small Enron 8-shot result was lost to an out-of-memory kill.

| dataset | body | shots/class | SetFit | Jev | gpt-5.4-mini | gpt-5.6-luna | train s | batch-1 p50 ms |
|---|---|---|---|---|---|---|---|---|
| ag_news | bge-small | 0 | 77 | 90 | 89 | 89 | 39 | – |
| ag_news | bge-small | 8 | 87 | 90 | 89 | 89 | 145 | torch 87, OpenVINO 40, OpenVINO int8 28 |
| enron_spam | bge-small | 0 | 57 | 98 | 99 | 99 | 24 | – |
| sst2 | bge-small | 0 | 87 | 96 | 95 | 96 | 18 | – |
| sst2 | bge-small | 8 | 89 | 96 | 95 | 96 | 27 | torch 64, OpenVINO 25, OpenVINO int8 17 |
| sst2 | bge-base | 8 | 91 | 96 | 95 | 96 | 84 | torch 197, OpenVINO 66, OpenVINO int8 40 |

OpenVINO and OpenVINO int8 gave the same accuracy as PyTorch on every run. For comparison, Jev took 0.8–0.9 s per call
in jev-eval, including the network round trip.
