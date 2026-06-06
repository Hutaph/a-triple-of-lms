# a-triple-of-lms

Benchmark multiple LMs on a small Big Data question set, then score the raw
answers with an LLM judge.

## Setup

Create `.env` from `.env.example` and set `OPENROUTER_API_KEY`.

```powershell
pip install -r requirements.txt
```

## Generate Raw Outputs

Run OpenRouter models:

```powershell
python src\openrouter_benchmark.py --model scout
python src\openrouter_benchmark.py --model maverick
python src\openrouter_benchmark.py --model qwen3-14b
python src\openrouter_benchmark.py --model qwen3-32b
python src\openrouter_benchmark.py --model qwen3-235b
```

Useful checks:

```powershell
python src\openrouter_benchmark.py --list-models
python src\openrouter_benchmark.py --model qwen3-14b --limit 1 --sleep 0
```

Raw outputs are written to:

```text
outputs/llama4/
outputs/qwen3/
outputs/phi3/
```

Each OpenRouter record includes model output, token usage, cost,
OpenRouter generation stats, timing, and token/s metrics.

## Run LLM Judge

After regenerating raw outputs, run:

```powershell
python src\llm_judge.py --provider openrouter --predictions outputs\llama4\llama4_scout_outputs.json outputs\llama4\llama4_maverick_outputs.json outputs\qwen3\Qwen3-14B_outputs.json outputs\qwen3\Qwen3-32B_outputs.json outputs\qwen3\Qwen3-235B-A22B-Thinking_outputs.json
```

Judge results include quality rubric scores, cost/latency efficiency metrics,
and reliability rates such as empty output, length stop, and missing generation
stats. Outputs are regenerated under:

```text
outputs/llm_judge/results/
outputs/llm_judge/tables/
outputs/llm_judge/visuals/
```
