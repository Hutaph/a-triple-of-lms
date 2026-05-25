"""
Benchmark Qwen3 models via OpenRouter API and Local LM Studio.

Models:
  - qwen3_4b: Local LM Studio (default: http://localhost:1234/v1)
  - qwen3_14b: OpenRouter API
  - qwen3_32b: OpenRouter API
  - qwen3_235b: OpenRouter API (Qwen3 235B A22B Thinking 2507)

Usage:
  python src/qwen3.py --model all
  python src/qwen3.py --model qwen3_4b
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen3"

DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 1500

SYSTEM_PROMPT = (
    "You are a senior Big Data engineer and Spark expert. "
    "Answer accurately, practically, and follow the user's instructions. "
    "For code tasks, provide correct and production-aware code."
)

DEFAULT_MODELS = {
    "qwen3_4b": {
        "name": "Qwen3-4B",
        "id": os.getenv("QWEN3_4B_MODEL", "qwen3-4b"), # LM Studio will map this or use loaded model
        "provider": "lm_studio"
    },
    "qwen3_14b": {
        "name": "Qwen3-14B",
        "id": os.getenv("QWEN3_14B_MODEL", "qwen/qwen-3-14b"),
        "provider": "openrouter"
    },
    "qwen3_32b": {
        "name": "Qwen3-32B",
        "id": os.getenv("QWEN3_32B_MODEL", "qwen/qwen-3-32b"),
        "provider": "openrouter"
    },
    "qwen3_235b": {
        "name": "Qwen3-235B-A22B-Thinking-2507",
        "id": os.getenv("QWEN3_235B_MODEL", "qwen/qwen3-235b-a22b-thinking-2507"),
        "provider": "openrouter"
    },
}


def load_environments() -> tuple[OpenAI | None, OpenAI]:
    """Load API clients for OpenRouter and LM Studio."""
    load_dotenv(PROJECT_ROOT / ".env")

    # OpenRouter client
    or_api_key = os.getenv("OPENROUTER_API_KEY")
    or_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not or_api_key:
        print("Warning: OPENROUTER_API_KEY is not set in .env. OpenRouter models will fail.")
        or_client = None
    else:
        or_client = OpenAI(base_url=or_base_url, api_key=or_api_key)

    # LM Studio client
    lm_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_client = OpenAI(base_url=lm_base_url, api_key="lm-studio")

    return or_client, lm_client


def load_samples(path: Path) -> list[dict]:
    """Load benchmark questions from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Benchmark JSON must be a list of samples.")

    return samples


def parse_generation_params(param_text: str | None) -> tuple[float, int]:
    """Parse temperature and max_tokens from string."""
    if not param_text:
        return DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS

    temp_match = re.search(r"temperature\s*=\s*([0-9.]+)", param_text)
    token_match = re.search(r"max_tokens\s*=\s*(\d+)", param_text)

    if temp_match:
        temperature = float(temp_match.group(1))

    if token_match:
        max_tokens = int(token_match.group(1))

    return temperature, max_tokens


def parse_thinking_content(response_text: str) -> tuple[str, str]:
    """Parse <think>...</think> blocks if present."""
    if not response_text:
        return "", ""

    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = response_text[think_match.end():].strip()
        return thinking, answer

    return "", response_text.strip()


def call_model(
    client: OpenAI,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, str, float]:
    """Call model via OpenAI API format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency = time.perf_counter() - start

    full_response = response.choices[0].message.content or ""
    thinking, answer = parse_thinking_content(full_response)

    return full_response, thinking, answer, latency


def run_generation_benchmark(
    or_client: OpenAI | None,
    lm_client: OpenAI,
    samples: list[dict],
    model_key: str,
    model_config: dict,
    sleep_seconds: float = 2.0,
) -> list[dict]:
    """Run generation (no auto-scoring) for a single model."""
    results = []
    model_name = model_config["name"]
    model_id = model_config["id"]
    provider = model_config["provider"]
    
    print(f"\nRunning generation for: {model_name} [{model_id}] via {provider}")
    
    if provider == "openrouter":
        if not or_client:
            print(f"Skipping {model_name} because OpenRouter API key is missing.")
            return []
        client = or_client
    else:
        client = lm_client

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        temperature, max_tokens = parse_generation_params(
            sample.get("recommended_generation_params")
        )

        started_at = datetime.now(timezone.utc).isoformat()

        try:
            full_response, thinking, answer, latency_s = call_model(
                client=client,
                model_id=model_id,
                prompt=sample["prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            error = None
        except Exception as e:
            full_response = None
            thinking = None
            answer = None
            latency_s = None
            error = str(e)
            print(f"  Error on {sample.get('sample_id')}: {error}")

        ended_at = datetime.now(timezone.utc).isoformat()

        record = {
            "sample_id": sample.get("sample_id"),
            "benchmark_scope": sample.get("benchmark_scope"),
            "category": sample.get("category"),
            "difficulty": sample.get("difficulty"),
            "topic": sample.get("topic"),
            "model_name": model_name,
            "model_id": model_id,
            "provider": provider,
            "prompt": sample.get("prompt"),
            "model_output": answer,
            "thinking_content": thinking,
            "full_response": full_response,
            "error": error,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "metadata": {
                "started_at": started_at,
                "ended_at": ended_at,
            },
        }

        results.append(record)
        time.sleep(sleep_seconds)

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Results saved to: {output_path}")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Big Data benchmark outputs for Qwen3 models (OpenRouter + LM Studio)."
    )
    parser.add_argument(
        "--model",
        choices=["qwen3_4b", "qwen3_14b", "qwen3_32b", "qwen3_235b", "all"],
        default="all",
        help="Which Qwen3 model to run.",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_FILE),
        help="Path to benchmark JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where benchmark outputs are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples. Use 0 for full dataset.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    or_client, lm_client = load_environments()
    samples = load_samples(Path(args.data))

    if args.limit:
        samples = samples[:args.limit]

    # Select models
    if args.model == "all":
        selected_models = DEFAULT_MODELS
    else:
        selected_models = {args.model: DEFAULT_MODELS[args.model]}

    for model_key, model_config in selected_models.items():
        results = run_generation_benchmark(
            or_client=or_client,
            lm_client=lm_client,
            samples=samples,
            model_key=model_key,
            model_config=model_config,
            sleep_seconds=args.sleep,
        )
        
        if results:
            fname = f"{safe_filename(model_config['name'])}_outputs.json"
            save_results(results, Path(args.output_dir) / fname)


if __name__ == "__main__":
    main()
