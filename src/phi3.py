import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phi3"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900

SYSTEM_PROMPT = (
    "You are a senior Big Data engineer and Spark expert. "
    "Answer accurately, practically, and follow the user's instructions. "
    "For code tasks, provide correct and production-aware code."
)

DEFAULT_MODELS = {
    "mini": (
        "Phi-3 Mini 4K",
        "PHI3_MINI_MODEL",
        "microsoft/Phi-3-mini-4k-instruct",
    ),
    "small": (
        "Phi-3 Small 8K",
        "PHI3_SMALL_MODEL",
        "microsoft/Phi-3-small-8k-instruct",
    ),
    "medium": (
        "Phi-3 Medium 4K",
        "PHI3_MEDIUM_MODEL",
        "microsoft/Phi-3-medium-4k-instruct",
    ),
}


def load_samples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

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


def model_registry() -> dict[str, dict[str, str]]:
    registry = {}
    for key, (display_name, env_name, default_model_id) in DEFAULT_MODELS.items():
        registry[key] = {
            "name": display_name,
            "id": os.getenv(env_name, default_model_id),
        }
    return registry


def normalize_hub_cache_dir(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None

    path = Path(cache_dir).expanduser()
    if path.name == "hub" and path.parent.exists() and any(path.parent.glob("models--*")):
        path = path.parent

    hub_path = path / "hub"
    if hub_path.exists() and any(hub_path.glob("models--*")):
        path = hub_path

    if path.name == "hub":
        os.environ.setdefault("HF_HOME", str(path.parent))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(path.parent / "transformers"))

    os.environ.setdefault("HF_HUB_CACHE", str(path))
    return str(path)


def torch_dtype_from_name(dtype_name: str, device: str):
    import torch

    if dtype_name == "auto":
        return torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def build_model_kwargs(
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
) -> dict:
    import torch

    dtype = torch_dtype_from_name(torch_dtype_name, device)
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit loading requires CUDA.")

        from transformers import BitsAndBytesConfig

        kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
        return kwargs

    kwargs["device_map"] = "auto" if device == "auto" else device
    return kwargs


def load_phi3_model(
    model_id_or_path: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        **build_model_kwargs(
            device=device,
            torch_dtype_name=torch_dtype_name,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        ),
    )
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"


def infer_input_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return getattr(model, "device", "cpu")


def call_phi3_model(
    tokenizer,
    model,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, float, int]:
    import torch

    text = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    device = infer_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    latency = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output, latency, len(generated_ids)


def expected_points_to_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [point.strip() for point in re.split(r";|\n", text) if point.strip()]


def score_output(sample: dict, output: str | None) -> dict[str, float]:
    if not output:
        return {}

    try:
        from benchmarks.spark_qa_benchmark import composite_score
    except Exception:
        return {}

    reference = sample.get("ground_truth") or ""
    must_have_points = expected_points_to_list(sample.get("expected_key_points"))
    scores = composite_score(output, reference, must_have_points)
    return {f"auto_{key}": value for key, value in scores.items()}


def build_result_record(
    sample: dict,
    model_name: str,
    model_id: str,
    output: str | None,
    error: str | None,
    temperature: float,
    max_tokens: int,
    latency_s: float | None,
    output_tokens: int | None,
    started_at: str,
    ended_at: str,
    include_scores: bool,
) -> dict:
    record = {
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
        "latency_s": round(latency_s, 3) if latency_s is not None else None,
        "output_tokens": output_tokens,
        "generation_params": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "metadata": {
            "started_at": started_at,
            "ended_at": ended_at,
        },
    }

    if include_scores and output is not None:
        record.update(score_output(sample, output))

    return record


def run_model_benchmark(
    samples: list[dict],
    model_name: str,
    model_id: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
    include_scores: bool,
    sleep_seconds: float,
) -> list[dict]:
    import torch

    print(f"Loading model: {model_name} ({model_id})")
    tokenizer, model = load_phi3_model(
        model_id_or_path=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        torch_dtype_name=torch_dtype_name,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )

    results = []
    print(f"Running benchmark for model: {model_name}")

    try:
        for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
            temperature, max_tokens = parse_generation_params(
                sample.get("recommended_generation_params")
            )
            started_at = datetime.now(timezone.utc).isoformat()

            try:
                output, latency_s, output_tokens = call_phi3_model(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=sample["prompt"],
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                )
                error = None
            except Exception as exc:
                output = None
                error = str(exc)
                latency_s = None
                output_tokens = None

            ended_at = datetime.now(timezone.utc).isoformat()
            results.append(
                build_result_record(
                    sample=sample,
                    model_name=model_name,
                    model_id=model_id,
                    output=output,
                    error=error,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_s=latency_s,
                    output_tokens=output_tokens,
                    started_at=started_at,
                    ended_at=ended_at,
                    include_scores=include_scores,
                )
            )

            if sleep_seconds:
                time.sleep(sleep_seconds)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def average(values: list[float | int | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def summarize(results: list[dict]) -> dict:
    models = sorted({result["model_name"] for result in results})
    summary = {}
    for model_name in models:
        rows = [result for result in results if result["model_name"] == model_name]
        summary[model_name] = {
            "count": len(rows),
            "errors": sum(1 for row in rows if row.get("error")),
            "avg_latency_s": average([row.get("latency_s") for row in rows]),
            "avg_output_tokens": average([row.get("output_tokens") for row in rows]),
            "avg_auto_score": average([row.get("auto_score") for row in rows]),
            "avg_auto_keyword_coverage": average(
                [row.get("auto_keyword_coverage") for row in rows]
            ),
            "avg_auto_instruction_following": average(
                [row.get("auto_instruction_following") for row in rows]
            ),
        }
    return summary


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Results saved to: {output_path}")


def save_all_outputs(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_results(results, output_dir / "phi3_benchmark_outputs.json")

    for model_name in sorted({result["model_name"] for result in results}):
        rows = [result for result in results if result["model_name"] == model_name]
        save_results(rows, output_dir / f"{safe_filename(model_name)}_outputs.json")

    summary = summarize(results)
    summary_path = output_dir / "phi3_benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Big Data benchmark for local Phi-3 models from Hugging Face cache."
    )
    parser.add_argument(
        "--model",
        choices=["mini", "small", "medium", "all"],
        default="all",
        help="Which Phi-3 model to run.",
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
        "--cache-dir",
        default=os.getenv("PHI3_HF_CACHE_DIR"),
        help="HF Hub cache dir. Use the folder containing models--*, or its parent hf_cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load models from local cache; do not download from Hugging Face.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device map for non-quantized loading.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used while loading the model.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load with bitsandbytes 4-bit quantization. Recommended for Colab T4.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if your cached Phi-3 variant requires remote code.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip automatic lexical scoring against ground_truth.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = normalize_hub_cache_dir(args.cache_dir)
    samples = load_samples(Path(args.data))
    if args.limit:
        samples = samples[: args.limit]

    registry = model_registry()
    selected = registry if args.model == "all" else {args.model: registry[args.model]}

    all_results = []
    for model_config in selected.values():
        results = run_model_benchmark(
            samples=samples,
            model_name=model_config["name"],
            model_id=model_config["id"],
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
            device=args.device,
            torch_dtype_name=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
            include_scores=not args.no_score,
            sleep_seconds=args.sleep,
        )
        all_results.extend(results)

    save_all_outputs(all_results, Path(args.output_dir))


if __name__ == "__main__":
    main()
