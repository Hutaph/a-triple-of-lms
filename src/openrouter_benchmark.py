import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 6000

SYSTEM_PROMPT = (
    "You are a senior Big Data engineer and Spark expert. "
    "Answer accurately, practically, and follow the user's instructions. "
    "For code tasks, provide correct and production-aware code."
    "After reasoning, always provide the final answer."
    "Never leave assistant content empty."
)

@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    family: str
    output_filename: str


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str


MODEL_GROUPS = {
    "llama4": ["llama4_scout", "llama4_maverick"],
    "qwen3": [
        "qwen3_4b",
        "qwen3_14b",
        "qwen3_32b",
        "qwen3_235b_a22b_thinking",
    ],
}

MODEL_ALIASES = {
    "scout": "llama4_scout",
    "llama4-scout": "llama4_scout",
    "llama4_scout": "llama4_scout",
    "maverick": "llama4_maverick",
    "llama4-maverick": "llama4_maverick",
    "llama4_maverick": "llama4_maverick",
    "qwen3-4b": "qwen3_4b",
    "qwen3_4b": "qwen3_4b",
    "qwen3-14b": "qwen3_14b",
    "qwen3_14b": "qwen3_14b",
    "qwen3-32b": "qwen3_32b",
    "qwen3_32b": "qwen3_32b",
    "qwen3-235b": "qwen3_235b_a22b_thinking",
    "qwen3-235b-a22b-thinking": "qwen3_235b_a22b_thinking",
    "qwen3_235b_a22b_thinking": "qwen3_235b_a22b_thinking",
}


def load_environment() -> tuple[OpenAI, OpenRouterConfig]:
    load_dotenv(PROJECT_ROOT / ".env")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)

    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Please add it to your .env file."
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    return client, OpenRouterConfig(api_key=api_key, base_url=base_url.rstrip("/"))


def build_model_registry() -> dict[str, ModelSpec]:
    return {
        "llama4_scout": ModelSpec(
            key="llama4_scout",
            display_name="Llama 4 Scout",
            model_id=os.getenv("LLAMA4_SCOUT_MODEL", "meta-llama/llama-4-scout"),
            family="llama4",
            output_filename="llama4_scout_outputs.json",
        ),
        "llama4_maverick": ModelSpec(
            key="llama4_maverick",
            display_name="Llama 4 Maverick",
            model_id=os.getenv("LLAMA4_MAVERICK_MODEL", "meta-llama/llama-4-maverick"),
            family="llama4",
            output_filename="llama4_maverick_outputs.json",
        ),
        "qwen3_4b": ModelSpec(
            key="qwen3_4b",
            display_name="Qwen3-4B",
            model_id=os.getenv("QWEN3_4B_MODEL", "qwen/qwen3-4b"),
            family="qwen3",
            output_filename="Qwen3-4B_outputs.json",
        ),
        "qwen3_14b": ModelSpec(
            key="qwen3_14b",
            display_name="Qwen3-14B",
            model_id=os.getenv("QWEN3_14B_MODEL", "qwen/qwen3-14b"),
            family="qwen3",
            output_filename="Qwen3-14B_outputs.json",
        ),
        "qwen3_32b": ModelSpec(
            key="qwen3_32b",
            display_name="Qwen3-32B",
            model_id=os.getenv("QWEN3_32B_MODEL", "qwen/qwen3-32b"),
            family="qwen3",
            output_filename="Qwen3-32B_outputs.json",
        ),
        "qwen3_235b_a22b_thinking": ModelSpec(
            key="qwen3_235b_a22b_thinking",
            display_name="Qwen3-235B-A22B-Thinking",
            model_id=os.getenv(
                "QWEN3_235B_A22B_THINKING_MODEL",
                "qwen/qwen3-235b-a22b-thinking-2507",
            ),
            family="qwen3",
            output_filename="Qwen3-235B-A22B-Thinking_outputs.json",
        ),
    }

def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if not isinstance(samples, list):
        raise ValueError("Benchmark JSON must be a list of samples.")

    return samples

def resolve_project_path(path_text: str | None, default: Path) -> Path:
    if not path_text:
        return default
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_model_selection(
    requested_models: list[str],
    registry: dict[str, ModelSpec],
) -> list[ModelSpec]:
    selected_keys = []

    for requested in requested_models:
        value = requested.strip()
        normalized = value.lower().replace("_", "-")

        if normalized == "all":
            keys = [key for group in MODEL_GROUPS.values() for key in group]
        elif normalized in MODEL_GROUPS:
            keys = MODEL_GROUPS[normalized]
        else:
            alias_key = MODEL_ALIASES.get(normalized) or MODEL_ALIASES.get(value)
            if not alias_key:
                valid = sorted(["all", *MODEL_GROUPS, *MODEL_ALIASES])
                raise ValueError(
                    f"Unknown model alias '{requested}'. Valid aliases: {', '.join(valid)}"
                )
            keys = [alias_key]

        for key in keys:
            if key not in registry:
                raise ValueError(f"Model '{key}' is not configured.")
            if key not in selected_keys:
                selected_keys.append(key)

    return [registry[key] for key in selected_keys]


def custom_model_spec(model_id: str, model_name: str | None, output_file: str | None) -> ModelSpec:
    display_name = model_name or model_id
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip("_") or "custom_model"
    return ModelSpec(
        key="custom",
        display_name=display_name,
        model_id=model_id,
        family="custom",
        output_filename=output_file or f"{safe_name}_outputs.json",
    )


def output_path_for_model(
    spec: ModelSpec,
    output_root: Path,
    output_dir: Path | None,
    output_file: str | None,
    selected_count: int,
) -> Path:
    if output_file and selected_count > 1:
        raise ValueError("--output-file can only be used when running one model.")

    base_dir = output_dir if output_dir else output_root / spec.family
    return base_dir / (output_file or spec.output_filename)


def print_available_models(registry: dict[str, ModelSpec]) -> None:
    print("Available model aliases:")
    for group_name, keys in MODEL_GROUPS.items():
        print(f"\n{group_name}:")
        for key in keys:
            spec = registry[key]
            print(f"  {key}: {spec.display_name} [{spec.model_id}]")

def parse_generation_params(param_text: str | None) -> tuple[float, int]:
    """
    Parse strings like:
    'temperature=0.2; max_tokens=6000'
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

def extract_response_usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        usage_data = usage.model_dump(mode="json")
    elif isinstance(usage, dict):
        usage_data = usage
    else:
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cost": getattr(usage, "cost", None),
        }

    return {
        "prompt_tokens": usage_data.get("prompt_tokens"),
        "completion_tokens": usage_data.get("completion_tokens"),
        "total_tokens": usage_data.get("total_tokens"),
        "cost": usage_data.get("cost"),
    }


def extract_choice_metadata(response) -> dict:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    return {
        "response_id": getattr(response, "id", None),
        "returned_model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        "native_finish_reason": getattr(choice, "native_finish_reason", None) if choice else None,
    }


def seconds_from_ms(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1000, 3)


def calculate_tokens_per_second(tokens: int | float | None, seconds: int | float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return round(float(tokens) / float(seconds), 3)


def fetch_generation_stats(
    config: OpenRouterConfig,
    generation_id: str | None,
    attempts: int,
    retry_sleep_s: float,
) -> tuple[dict | None, str | None]:
    if not generation_id:
        return None, "Missing generation id."

    headers = {"Authorization": f"Bearer {config.api_key}"}
    url = f"{config.base_url}/generation"
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = httpx.get(
                url,
                headers=headers,
                params={"id": generation_id},
                timeout=30,
            )
            if response.status_code == 404 and attempt < attempts:
                time.sleep(retry_sleep_s)
                continue
            response.raise_for_status()
            data = response.json().get("data") or {}
            return normalize_generation_stats(data), None
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(retry_sleep_s)

    return None, last_error


def normalize_generation_stats(data: dict) -> dict:
    return {
        "id": data.get("id"),
        "api_type": data.get("api_type"),
        "model": data.get("model"),
        "provider_name": data.get("provider_name"),
        "router": data.get("router"),
        "service_tier": data.get("service_tier"),
        "finish_reason": data.get("finish_reason"),
        "native_finish_reason": data.get("native_finish_reason"),
        "created_at": data.get("created_at"),
        "latency_ms": data.get("latency"),
        "generation_time_ms": data.get("generation_time"),
        "moderation_latency_ms": data.get("moderation_latency"),
        "latency_s": seconds_from_ms(data.get("latency")),
        "generation_time_s": seconds_from_ms(data.get("generation_time")),
        "moderation_latency_s": seconds_from_ms(data.get("moderation_latency")),
        "tokens_prompt": data.get("tokens_prompt"),
        "tokens_completion": data.get("tokens_completion"),
        "tokens_total": total_if_present(data.get("tokens_prompt"), data.get("tokens_completion")),
        "native_tokens_prompt": data.get("native_tokens_prompt"),
        "native_tokens_completion": data.get("native_tokens_completion"),
        "native_tokens_reasoning": data.get("native_tokens_reasoning"),
        "native_tokens_cached": data.get("native_tokens_cached"),
        "total_cost": data.get("total_cost"),
        "usage": data.get("usage"),
        "upstream_inference_cost": data.get("upstream_inference_cost"),
        "is_byok": data.get("is_byok"),
        "streamed": data.get("streamed"),
        "num_fetches": data.get("num_fetches"),
        "request_id": data.get("request_id"),
        "upstream_id": data.get("upstream_id"),
    }


def total_if_present(left: int | float | None, right: int | float | None) -> int | float | None:
    if left is None or right is None:
        return None
    return left + right


def call_openrouter_model(
    client: OpenAI,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict | None, dict]:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output = response.choices[0].message.content

    return output, extract_response_usage(response), extract_choice_metadata(response)

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
    latency_s: float,
    usage: dict | None,
    response_metadata: dict | None = None,
    generation_stats: dict | None = None,
    generation_stats_error: str | None = None,
    generation_stats_lookup_s: float | None = None,
) -> dict:
    completion_tokens = None
    tokens_per_second = None
    tokens_per_second_generation = None
    tokens_per_second_openrouter_latency = None
    native_tokens_per_second_generation = None
    native_tokens_per_second_openrouter_latency = None

    if usage:
        completion_tokens = usage.get("completion_tokens")

    generation_completion_tokens = None
    native_generation_completion_tokens = None
    generation_time_s = None
    openrouter_latency_s = None
    moderation_latency_s = None

    if generation_stats:
        generation_completion_tokens = generation_stats.get("tokens_completion")
        native_generation_completion_tokens = generation_stats.get("native_tokens_completion")
        generation_time_s = generation_stats.get("generation_time_s")
        openrouter_latency_s = generation_stats.get("latency_s")
        moderation_latency_s = generation_stats.get("moderation_latency_s")

    if completion_tokens is not None and latency_s > 0:
        tokens_per_second = calculate_tokens_per_second(completion_tokens, latency_s)

    tokens_per_second_generation = calculate_tokens_per_second(
        generation_completion_tokens or completion_tokens,
        generation_time_s,
    )
    tokens_per_second_openrouter_latency = calculate_tokens_per_second(
        generation_completion_tokens or completion_tokens,
        openrouter_latency_s,
    )
    native_tokens_per_second_generation = calculate_tokens_per_second(
        native_generation_completion_tokens or completion_tokens,
        generation_time_s,
    )
    native_tokens_per_second_openrouter_latency = calculate_tokens_per_second(
        native_generation_completion_tokens or completion_tokens,
        openrouter_latency_s,
    )

    return {
        "sample_id": sample.get("sample_id"),
        "benchmark_scope": sample.get("benchmark_scope"),
        "category": sample.get("category"),
        "difficulty": sample.get("difficulty"),
        "topic": sample.get("topic"),
        "model_name": model_name,
        "model_id": model_id,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": sample.get("prompt"),
        "model_output": output,
        "error": error,
        "status": "success" if error is None else "failed",
        "generation_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "usage": usage,
        "generation_stats": generation_stats,
        "metrics": {
            "latency_s": latency_s,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_client": tokens_per_second,
            "tokens_per_second_generation": tokens_per_second_generation,
            "tokens_per_second_openrouter_latency": tokens_per_second_openrouter_latency,
            "native_tokens_per_second_generation": native_tokens_per_second_generation,
            "native_tokens_per_second_openrouter_latency": native_tokens_per_second_openrouter_latency,
            "output_char_count": len(output) if output else 0,
            "output_word_count": len(output.split()) if output else 0,
        },
        "timing": {
            "client_total_s": latency_s,
            "openrouter_latency_s": openrouter_latency_s,
            "generation_time_s": generation_time_s,
            "moderation_latency_s": moderation_latency_s,
            "generation_stats_lookup_s": generation_stats_lookup_s,
        },
        "metadata": {
            "started_at": started_at,
            "ended_at": ended_at,
            **(response_metadata or {}),
            "generation_stats_error": generation_stats_error,
        },
    }

def run_model_benchmark(
    client: OpenAI,
    config: OpenRouterConfig,
    samples: list[dict],
    model_name: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    fetch_stats: bool = True,
    generation_stats_attempts: int = 20,
    generation_stats_retry_s: float = 1.5,
    sleep_seconds: float = 1.0,
) -> list[dict]:
    results = []

    print(f"Running benchmark for model: {model_name} ({model_id})")

    for sample in tqdm(samples, desc=f"Running {model_name}"):
        started_at = datetime.now(timezone.utc).isoformat()
        start_timer = time.perf_counter()
        generation_stats = None
        generation_stats_error = None
        generation_stats_lookup_s = None

        try:
            output, usage, response_metadata = call_openrouter_model(
                client=client,
                model_id=model_id,
                prompt=sample["prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_s = round(time.perf_counter() - start_timer, 3)
            error = None

            if fetch_stats:
                stats_timer = time.perf_counter()
                generation_stats, generation_stats_error = fetch_generation_stats(
                    config=config,
                    generation_id=response_metadata.get("response_id"),
                    attempts=generation_stats_attempts,
                    retry_sleep_s=generation_stats_retry_s,
                )
                generation_stats_lookup_s = round(time.perf_counter() - stats_timer, 3)
        except Exception as e:
            latency_s = round(time.perf_counter() - start_timer, 3)
            output = None
            usage = None
            response_metadata = None
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
            latency_s=latency_s,
            usage=usage,
            response_metadata=response_metadata,
            generation_stats=generation_stats,
            generation_stats_error=generation_stats_error,
            generation_stats_lookup_s=generation_stats_lookup_s,
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
        description="Run Big Data benchmark for OpenRouter models such as Llama 4 and Qwen3."
    )

    parser.add_argument(
        "--model",
        nargs="+",
        default=["llama4"],
        help=(
            "Model alias(es) to run. Examples: scout, maverick, llama4, "
            "qwen3-14b, qwen3-32b, qwen3-235b, qwen3, all."
        ),
    )
    parser.add_argument(
        "--model-id",
        help="Run one custom OpenRouter model ID instead of a configured alias.",
    )
    parser.add_argument(
        "--model-name",
        help="Display name for --model-id. Defaults to the model ID.",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_FILE),
        help="Benchmark JSON file. Defaults to data/bigdata_10_questions.json.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root output directory. Defaults to outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory for all selected models.",
    )
    parser.add_argument(
        "--output-file",
        help="Override output file name. Only valid when running one model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum output tokens per request. Default: {DEFAULT_MAX_TOKENS}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N samples. Useful for smoke tests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--skip-generation-stats",
        action="store_true",
        help="Skip OpenRouter /generation metadata lookup after each completion.",
    )
    parser.add_argument(
        "--generation-stats-attempts",
        type=int,
        default=20,
        help="Number of attempts for OpenRouter /generation metadata lookup. Default: 20.",
    )
    parser.add_argument(
        "--generation-stats-retry-s",
        type=float,
        default=1.5,
        help="Seconds between /generation metadata lookup retries. Default: 1.5.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print configured model aliases and exit.",
    )

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    registry = build_model_registry()

    if args.list_models:
        print_available_models(registry)
        return

    if args.model_id:
        selected_models = [
            custom_model_spec(
                model_id=args.model_id,
                model_name=args.model_name,
                output_file=args.output_file,
            )
        ]
    else:
        selected_models = resolve_model_selection(args.model, registry)

    data_file = resolve_project_path(args.data, DATA_FILE)
    output_root = resolve_project_path(args.output_root, OUTPUT_ROOT)
    output_dir = resolve_project_path(args.output_dir, OUTPUT_ROOT) if args.output_dir else None

    client, config = load_environment()
    samples = load_samples(data_file)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be >= 0.")
        samples = samples[: args.limit]
    if args.generation_stats_attempts < 1:
        raise ValueError("--generation-stats-attempts must be >= 1.")
    if args.generation_stats_retry_s < 0:
        raise ValueError("--generation-stats-retry-s must be >= 0.")

    for spec in selected_models:
        results = run_model_benchmark(
            client=client,
            config=config,
            samples=samples,
            model_name=spec.display_name,
            model_id=spec.model_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            fetch_stats=not args.skip_generation_stats,
            generation_stats_attempts=args.generation_stats_attempts,
            generation_stats_retry_s=args.generation_stats_retry_s,
            sleep_seconds=args.sleep,
        )

        output_path = output_path_for_model(
            spec=spec,
            output_root=output_root,
            output_dir=output_dir,
            output_file=args.output_file,
            selected_count=len(selected_models),
        )
        save_results(results, output_path)

if __name__ == "__main__":
    main()
