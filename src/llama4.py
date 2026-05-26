import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llama4"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900

def load_environment() -> tuple[OpenAI, dict]:
    load_dotenv(PROJECT_ROOT / ".env")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")

    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Please add it to your .env file."
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    models = {
        "llama4_scout": os.getenv(
            "LLAMA4_SCOUT_MODEL",
            "meta-llama/llama-4-scout"
        ),
        "llama4_maverick": os.getenv(
            "LLAMA4_MAVERICK_MODEL",
            "meta-llama/llama-4-maverick"
        ),
    }

    return client, models

def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        raise ValueError("Benchmark JSON must be a list of samples.")

    return samples

def parse_generation_params(param_text: str | None) -> tuple[float, int]:
    """
    Parse strings like:
    'temperature=0.2; max_tokens=400'
    """
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

def call_openrouter_model(
        client: OpenAI,
        model_id: str,
        prompt: str,
        temperature: float,
        max_tokens: int
) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Big Data engineer and Spark expert. "
                    "Answer accurately, practically, and follow the user's instructions. "
                    "For code tasks, provide correct and production-aware code."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

def build_result_record(
    sample: dict,
    model_name: str,
    model_id: str,
    output: str | None,
    error: str | None,
    temperature: float,
    max_tokens: int,
    started_at: str,
    ended_at: str,
) -> dict:
    return {
        "sample_id": sample.get("sample_id"),
        "benchmark_scope": sample.get("benchmark_scope"),
        "category": sample.get("category"),
        "difficulty": sample.get("difficulty"),
        "topic": sample.get("topic"),
        "model_name": model_name,
        "model_id": model_id,
        "prompt": sample.get("prompt"),
        "model_output": output,
        "error": error,
        "generation_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "metadata": {
            "started_at": started_at,
            "ended_at": ended_at,
        },
    }

def run_model_benchmark(
    client: OpenAI,
    samples: list[dict],
    model_name: str,
    model_id: str,
    sleep_seconds: float = 1.0,
) -> list[dict]:
    results = []

    print(f"Running benchmark for model: {model_name} ({model_id})")

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        temperature, max_tokens = parse_generation_params(sample.get("recommended_generation_params"))

        started_at = datetime.now(timezone.utc).isoformat()

        try:
            output = call_openrouter_model(
                client=client,
                model_id=model_id,
                prompt=sample["prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            error = None
        except Exception as e:
            output = None
            error = str(e)
        
        ended_at = datetime.now(timezone.utc).isoformat()

        record = build_result_record(
            sample=sample,
            model_name=model_name,
            model_id=model_id,
            output=output,
            error=error,
            temperature=temperature,
            max_tokens=max_tokens,
            started_at=started_at,
            ended_at=ended_at,
        )

        results.append(record)

        time.sleep(sleep_seconds)

    return results

def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Big Data benchmark for Llama 4 Scout and Maverick via OpenRouter."
    )

    parser.add_argument(
        "--model",
        choices=["scout", "maverick", "all"],
        default="all",
        help="Which Llama 4 model to run.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )

    args = parser.parse_args()

    client, models = load_environment()
    samples = load_samples(DATA_FILE)

    selected_models = {}

    if args.model in ["scout", "all"]:
        selected_models["llama4_scout"] = {
            "display_name": "Llama 4 Scout",
            "model_id": models["llama4_scout"],
        }

    if args.model in ["maverick", "all"]:
        selected_models["llama4_maverick"] = {
            "display_name": "Llama 4 Maverick",
            "model_id": models["llama4_maverick"],
        }

    for model_key, model_info in selected_models.items():
        results = run_model_benchmark(
            client=client,
            samples=samples,
            model_name=model_info["display_name"],
            model_id=model_info["model_id"],
            sleep_seconds=args.sleep,
        )

        output_path = OUTPUT_DIR / f"{model_key}_outputs.json"
        save_results(results, output_path)

if __name__ == "__main__":
    main()
    