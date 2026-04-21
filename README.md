# LLMEvalKit

Universal LLM evaluation harness. Point it at **any model**, run **any benchmark**, get a scorecard.

## Why this exists

Every serious LLM eval framework (lm-evaluation-harness, lighteval, evalchemy, Inspect AI, BFCL, SWE-bench, RULER, HarmBench…) speaks **OpenAI-compatible HTTP**. Every modern inference server (vLLM, llama.cpp, SGLang, TGI, Ollama, LM Studio) **exposes** an OpenAI-compatible endpoint. That's the universal glue.

LLMEvalKit owns the glue:

```
  Model                    Adapter                 Runner                  Benchmark
  ─────────               ────────────            ────────────           ──────────────
  GGUF file       ───▶   llama-server      ───▶   lm-eval-harness  ───▶  MMLU-Pro, GSM8K, …
  HF model        ───▶   vllm serve        ───▶   evalchemy        ───▶  AIME, MATH-500, LCB
  OpenAI URL      ───▶   passthrough       ───▶   BFCL             ───▶  tool calling
  Anthropic       ───▶   passthrough       ───▶   mini-swe-agent   ───▶  SWE-bench Verified
  Custom PyTorch  ───▶   FastAPI shim      ───▶   RULER            ───▶  long context
  (finetunes,             (template given)        HarmBench              jailbreak ASR
   frankenmerges,                                 Arena-Hard             chat quality
   novel archs)                                   Inspect AI             200+ more evals
```

One entry point. One YAML config per model. One YAML profile per benchmark bundle.

## Install

```bash
git clone https://github.com/Nodenester/LLMEvalKit.git
cd LLMEvalKit
pip install -e .
```

Then install only the eval frameworks you need:

```bash
python scripts/install.py --list
python scripts/install.py --framework lm-evaluation-harness
python scripts/install.py --framework evalchemy
python scripts/install.py --framework inspect-ai         # pip-only
python scripts/install.py --framework gorilla            # BFCL lives here
python scripts/install.py --all                          # everything (hours + GBs)
```

Frameworks are cloned under `frameworks/` (gitignored). You can delete / reinstall individually.

## Usage

```bash
# List what's available
python -m llmevalkit list-profiles
python -m llmevalkit list-runners
python -m llmevalkit list-adapters

# Dry-run to see the plan
python -m llmevalkit run --model configs/examples/openai_gpt4o.yaml --profile quick --dry-run

# Actually evaluate
python -m llmevalkit run --model configs/examples/local_llamacpp_gguf.yaml --profile coding
python -m llmevalkit run --model configs/examples/vllm_hf.yaml --profile full
```

Results land in `results/<model>__<profile>__<timestamp>/` with a markdown `scorecard.md` + per-runner JSON.

## Built-in profiles

| profile      | what                                         | time         |
|--------------|----------------------------------------------|--------------|
| `quick`      | 5-min sanity + GSM8K / ARC subsets           | ~5 min       |
| `knowledge`  | MMLU, MMLU-Pro, ARC, TruthfulQA, DROP        | ~30 min      |
| `reasoning`  | MMLU-Pro, BBH, GPQA-Diamond, IFEval, SimpleQA| ~1 h         |
| `math`       | GSM8K, MATH-500, AIME24/25                   | ~1 h         |
| `coding`     | HumanEval+, MBPP+, BigCodeBench, LiveCodeBench | ~2-4 h     |
| `agentic`    | BFCL, SWE-bench Verified, GAIA               | many hours   |
| `multilingual`| MGSM, XNLI, Global-MMLU                     | ~1 h         |
| `longctx`    | RULER up to 128K, NIAH                       | ~2-6 h       |
| `safety`     | HarmBench, JailbreakBench                    | ~1-2 h       |
| `chat`       | Arena-Hard-Auto (needs judge API key)        | ~30 min      |
| `full`       | everything                                   | many hours   |

Profiles are plain YAML — copy one from `llmevalkit/profiles/` and edit.

## Model config examples

Existing OpenAI-compatible URL (fastest, zero overhead):
```yaml
name: my-model
kind: openai
base_url: http://127.0.0.1:8787   # llama-server / vLLM / SGLang / TGI / Ollama already running
model_name: my-model
```

Local GGUF (starts llama-server for you):
```yaml
name: my-gguf
kind: gguf
model_path: /path/to/your-model-Q4_K_M.gguf
model_name: my-gguf
ctx_size: 32768
n_gpu_layers: 999
```

HuggingFace model (starts vLLM for you):
```yaml
name: qwen
kind: vllm
model: Qwen/Qwen3.5-9B-Instruct
max_model_len: 32768
tensor_parallel_size: 1
```

Anthropic / OpenAI hosted:
```yaml
name: claude
kind: anthropic
model_name: claude-opus-4-7
# needs ANTHROPIC_API_KEY in env
```

**Custom PyTorch model** (finetunes, frankenmerges, novel architectures, 1-bit/ES-trained weights — anything that doesn't fit a standard inference server):
1. Copy `scripts/custom_model_server_template.py` to your project, e.g. `/path/to/your/project/serve.py`.
2. Fill in `load_model()` and `generate()` — load your weights, run your forward.
3. Point a config at it:
   ```yaml
   name: my-custom
   kind: custom
   script: /path/to/your/project/serve.py
   model_name: my-custom-v1
   port: 8001
   env:
       CUDA_VISIBLE_DEVICES: "0"
   ```
4. Run any profile. The kit starts your FastAPI shim, waits for `/v1/models`, points every runner at it, shuts it down at the end.

## Architecture

```
llmevalkit/
├── adapters/         # turn anything into an OpenAI-compatible URL
├── runners/          # each wraps one external eval framework
├── profiles/         # YAML bundles of runners + tasks
├── results/          # scorecard + summary writing
└── cli.py            # typer app, single entrypoint
scripts/
├── install.py        # clone + pip-install external frameworks on demand
└── custom_model_server_template.py   # FastAPI shim for custom PyTorch models
configs/examples/     # one YAML per model kind
frameworks/           # cloned eval repos (gitignored)
results/              # run outputs (gitignored)
```

Adding a new runner: subclass `Runner`, implement `run(endpoint, tasks, output_dir)`, register in `runners/__init__.py`.

Adding a new adapter: subclass `ModelAdapter`, implement `start()` → `Endpoint`, register in `adapters/__init__.py`.

## Status

Scaffold complete. Below is the verified cross-platform status matrix (tested on Windows 11 + RTX 5060 Ti, Python 3.12, against a local Qwen3.5-0.8B shim).

| runner | Linux/Docker | Windows | notes |
|---|---|---|---|
| `simple-chat` | ✓ | ✓ | Stdlib httpx, no extras |
| `lm-eval-harness` | ✓ | ✓ | Works directly against chat-completions endpoint |
| `inspect-ai` | ✓ | ✓ | 200+ evals via `inspect_evals/<task>`. `ifeval` needs `pip install git+https://github.com/josejg/instruction_following_eval` |
| `bigcodebench` | ✓ | partial | Generation works; evaluate needs `e2b` sandbox |
| `evalplus` | ✓ | ✗ | Upstream uses `signal.alarm` (Unix-only) |
| `lighteval` | ✓ | ✗ | Writes cache paths with `|` char — fs reject |
| `evalchemy` | ✓ | ✗ | Custom lm-eval fork fails MAX_PATH; vllm fails path chars |
| `bfcl` | ✓ | ✓* | *Requires model_name in BFCL's fixed registry; use `inspect_evals/bfcl` for arbitrary endpoints |
| `mini-swe-agent` | ✓ | ✓* | *v2.x entrypoints moved; use `inspect_evals/swe_bench`. SWE-bench always needs Docker |
| `ruler` | ✓ (WSL) | ✗ | Bash + vllm/TRT-LLM; use `inspect_evals/niah` for cross-platform NIAH |
| `harmbench` | ✓ | ✗ | Requires vllm + legacy spacy pin; use `inspect_evals/jailbreakbench` |
| `arena-hard` | ✓ | ✓* | *Needs judge model (GPT-4o) API key to score |

**Bottom line on Windows**: `simple-chat` + `lm-eval-harness` + `inspect-ai` (~200 of its evals) + `bigcodebench`-generation cover nearly every benchmark you'd want. For the rest, WSL or Docker.

Verification command (against any running OpenAI-compatible endpoint):

```bash
python -m llmevalkit run --model configs/examples/local_llamaserver_existing.yaml --profile windows_demo
```

Example scorecard output:

```
# Scorecard: qwen3.5-0.8b-running
Profile: windows_demo
Wall time: 7.9 min
Runners OK: 4/4

[PASS] simple-chat       0.1 min   prompts_ok: 6/6
[PASS] lm-eval-harness   2.0 min   gsm8k flexible-extract: 0.10
[PASS] inspect-ai        4.6 min   gsm8k / humaneval / mbpp / ifeval — all scored
[PASS] bigcodebench      1.2 min   1 sample generated (eval needs e2b)
```
