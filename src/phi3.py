import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phi3"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900
DEFAULT_SMALL_ONNX_MODEL_ID = "microsoft/Phi-3-small-8k-instruct-onnx-cuda"
DEFAULT_SMALL_ONNX_VARIANT = "cuda-int4-rtn-block-32"
DEFAULT_SMALL_ONNX_EXECUTION_PROVIDER = "cuda"

SYSTEM_PROMPT = (
    "You are a senior Big Data engineer and Spark expert. "
    "Answer accurately, practically, and follow the user's instructions. "
    "For code tasks, provide correct and production-aware code."
)

DEFAULT_MODELS = {
    "mini": {
        "name": "Phi-3 Mini 4K",
        "env": "PHI3_MINI_MODEL",
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": False,
    },
    "small": {
        "name": "Phi-3 Small 8K",
        "env": "PHI3_SMALL_MODEL",
        "id": "microsoft/Phi-3-small-8k-instruct",
        "onnx_env": "PHI3_SMALL_ONNX_MODEL",
        "onnx_id": DEFAULT_SMALL_ONNX_MODEL_ID,
        "trust_remote_code": True,
    },
    "medium": {
        "name": "Phi-3 Medium 4K",
        "env": "PHI3_MEDIUM_MODEL",
        "id": "microsoft/Phi-3-medium-4k-instruct",
        "trust_remote_code": False,
    },
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


def model_registry() -> dict[str, dict[str, Any]]:
    registry = {}
    for key, config in DEFAULT_MODELS.items():
        registry[key] = {
            "name": config["name"],
            "id": os.getenv(config["env"], config["id"]),
            "onnx_id": os.getenv(config.get("onnx_env", ""), config.get("onnx_id", "")),
            "trust_remote_code": config["trust_remote_code"],
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


def quantized_device_map(device: str):
    import torch

    if device == "cpu":
        raise ValueError("4-bit loading requires CUDA.")
    if torch.cuda.device_count() == 1:
        return {"": 0}
    return "auto"


def bnb_compute_dtype(dtype):
    import torch

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    return dtype


def build_model_kwargs(
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
) -> dict:
    import torch

    dtype = torch_dtype_from_name(torch_dtype_name, device)
    kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit loading requires CUDA.")

        from transformers import BitsAndBytesConfig

        kwargs.update(
            {
                "device_map": quantized_device_map(device),
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=bnb_compute_dtype(dtype),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
        return kwargs

    kwargs["device_map"] = "auto" if device == "auto" else device
    return kwargs


def sanitize_phi3_config(config):
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and not rope_scaling:
        config.rope_scaling = None

    if getattr(config, "model_type", None) == "phi3small":
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "short_factor" not in rope_scaling:
            config.rope_scaling = None

    return config


def model_load_hints(exc: Exception, model_id_or_path: str) -> str:
    message = str(exc)
    hints = []
    model_ref = model_id_or_path.lower()

    if "trust_remote_code" in message:
        hints.append("Phi-3 Small uses Hugging Face custom code; retry with --trust-remote-code.")
    if "unsupported scalartype" in message.lower():
        hints.append(
            "If this is Phi-3 Small on Colab T4, the model's custom BF16/Triton path is not "
            "T4-friendly; try an A100/A6000/H100 runtime or the ONNX CUDA variant."
        )
    if "flash attention" in message.lower() or "flash_attn" in message.lower():
        hints.append("Phi-3 Small requires flash-attention-capable GPU/runtime for its dense attention layers.")
    if "onnxruntime_genai" in message.lower():
        hints.append("Install ONNX Runtime GenAI CUDA with: pip install --pre onnxruntime-genai-cuda.")
    if "load model from" in message.lower() and ".onnx" in message.lower():
        hints.append(
            "The ONNX cache may be incomplete. Re-download the variant with "
            "hf download ... --include '<variant>/*' --local-dir <dir>, then pass --small-onnx-dir."
        )
    if "cuda" in message.lower() and ("provider" in message.lower() or "library" in message.lower()):
        hints.append("Check that onnxruntime-genai-cuda can see CUDA/cuDNN in this runtime.")
    if ("phi-3-small" in model_ref or "phi3small" in message.lower()) and "onnx" not in model_ref:
        hints.append("Phi-3 Small is the only model in this script that uses custom Triton/block-sparse code.")

    return (" Hint: " + " ".join(dict.fromkeys(hints))) if hints else ""


def load_phi3_model(
    model_id_or_path: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(torch_dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    config = sanitize_phi3_config(config)
    config.torch_dtype = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        cache_dir=cache_dir,
        config=config,
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


def resolve_onnx_model_dir(
    onnx_model_id: str,
    onnx_variant: str,
    onnx_model_dir: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> Path:
    if onnx_model_dir:
        model_dir = Path(onnx_model_dir).expanduser()
        if model_dir.name != onnx_variant and (model_dir / onnx_variant).exists():
            model_dir = model_dir / onnx_variant
        return model_dir

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=onnx_model_id,
            allow_patterns=[f"{onnx_variant}/*"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    return snapshot_dir / onnx_variant


def read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 4096:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:256]
    except OSError:
        return False
    return "version https://git-lfs.github.com/spec" in head or "oid sha256:" in head


def download_hint(onnx_model_id: str, onnx_variant: str) -> str:
    return (
        f"Download a materialized copy with: hf download {onnx_model_id} "
        f"--include {onnx_variant}/* --local-dir <dir>; then run with "
        f"--small-onnx-dir <dir>/{onnx_variant}"
    )


def validate_onnx_model_dir(model_dir: Path, onnx_model_id: str, onnx_variant: str) -> None:
    genai_config = model_dir / "genai_config.json"
    if not genai_config.exists():
        raise FileNotFoundError(
            f"ONNX GenAI config not found: {genai_config}. "
            f"{download_hint(onnx_model_id, onnx_variant)}"
        )

    config = read_json_file(genai_config)
    decoder = config.get("model", {}).get("decoder", {})
    onnx_filename = decoder.get("filename")
    if not isinstance(onnx_filename, str) or not onnx_filename:
        raise ValueError(f"Missing model.decoder.filename in {genai_config}")

    onnx_path = model_dir / onnx_filename
    external_data_path = model_dir / f"{onnx_filename}.data"
    missing_paths = [
        str(path)
        for path in (onnx_path, external_data_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Incomplete ONNX model folder. Missing: "
            + ", ".join(missing_paths)
            + ". "
            + download_hint(onnx_model_id, onnx_variant)
        )

    pointer_paths = [
        str(path)
        for path in (onnx_path, external_data_path)
        if is_lfs_pointer(path)
    ]
    if pointer_paths:
        raise RuntimeError(
            "ONNX files are Git LFS pointer files, not real model weights: "
            + ", ".join(pointer_paths)
            + ". "
            + download_hint(onnx_model_id, onnx_variant)
        )

    if external_data_path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(
            f"ONNX external data file looks incomplete: {external_data_path} "
            f"({external_data_path.stat().st_size} bytes). "
            f"{download_hint(onnx_model_id, onnx_variant)}"
        )


def load_phi3_onnx_model(
    onnx_model_id: str,
    onnx_variant: str,
    onnx_model_dir: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    execution_provider: str,
):
    import onnxruntime_genai as og

    model_dir = resolve_onnx_model_dir(
        onnx_model_id=onnx_model_id,
        onnx_variant=onnx_variant,
        onnx_model_dir=onnx_model_dir,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    validate_onnx_model_dir(model_dir, onnx_model_id, onnx_variant)

    config = og.Config(str(model_dir))
    if execution_provider != "follow_config":
        config.clear_providers()
        if execution_provider != "cpu":
            config.append_provider(execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    return tokenizer, model, str(model_dir)


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


def build_onnx_chat_prompt(prompt: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        "<|assistant|>\n"
    )


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
) -> tuple[str, float, int, int]:
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
    return output, latency, int(inputs["input_ids"].shape[-1]), len(generated_ids)


def call_phi3_onnx_model(
    tokenizer,
    model,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, float, int, int]:
    import onnxruntime_genai as og

    text = build_onnx_chat_prompt(prompt)
    input_tokens = tokenizer.encode(text)

    params = og.GeneratorParams(model)
    search_options = {
        "max_length": len(input_tokens) + max_new_tokens,
        "min_length": len(input_tokens),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        search_options["temperature"] = temperature
    params.set_search_options(**search_options)

    generator = og.Generator(model, params)
    stream = tokenizer.create_stream()
    output_chunks = []
    output_tokens = 0

    start = time.perf_counter()
    try:
        generator.append_tokens(input_tokens)
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            output_chunks.append(stream.decode(new_token))
            output_tokens += 1
    finally:
        del generator

    latency = time.perf_counter() - start
    return "".join(output_chunks).strip(), latency, len(input_tokens), output_tokens


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
    started_at: str,
    ended_at: str,
    latency_s: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    include_scores: bool,
) -> dict:
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    tokens_per_second = None
    if completion_tokens is not None and latency_s and latency_s > 0:
        tokens_per_second = round(completion_tokens / latency_s, 3)

    output_text = output or ""
    record = {
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
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "metrics": {
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "tokens_per_second": tokens_per_second,
            "output_char_count": len(output_text),
            "output_word_count": len(output_text.split()),
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
    onnx_model_id: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    torch_dtype_name: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
    backend: str,
    onnx_variant: str,
    onnx_model_dir: str | None,
    onnx_execution_provider: str,
    include_scores: bool,
    sleep_seconds: float,
    show_traceback: bool,
) -> list[dict]:
    import torch

    active_model_id = onnx_model_id if backend == "onnx" else model_id
    print(f"Loading model: {model_name} ({active_model_id}); backend={backend}; trust_remote_code={trust_remote_code}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if backend == "onnx":
            tokenizer, model, resolved_model_dir = load_phi3_onnx_model(
                onnx_model_id=onnx_model_id,
                onnx_variant=onnx_variant,
                onnx_model_dir=onnx_model_dir,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                execution_provider=onnx_execution_provider,
            )
            active_model_id = f"{onnx_model_id}/{onnx_variant}"
            print(f"ONNX model dir: {resolved_model_dir}; execution_provider={onnx_execution_provider}")
        else:
            tokenizer, model = load_phi3_model(
                model_id_or_path=model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                device=device,
                torch_dtype_name=torch_dtype_name,
                load_in_4bit=load_in_4bit,
                trust_remote_code=trust_remote_code,
            )
    except Exception as exc:
        if show_traceback:
            traceback.print_exc()
        error = f"Model load failed: {exc}{model_load_hints(exc, active_model_id)}"
        print(error)
        results = []
        for sample in samples:
            temperature, max_tokens = parse_generation_params(
                sample.get("recommended_generation_params")
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            results.append(
                build_result_record(
                    sample=sample,
                    model_name=model_name,
                    model_id=active_model_id,
                    output=None,
                    error=error,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    started_at=timestamp,
                    ended_at=timestamp,
                    latency_s=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    include_scores=include_scores,
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    results = []
    print(f"Running benchmark for model: {model_name}")

    try:
        for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
            temperature, max_tokens = parse_generation_params(
                sample.get("recommended_generation_params")
            )
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                if backend == "onnx":
                    output, latency_s, prompt_tokens, completion_tokens = call_phi3_onnx_model(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=sample["prompt"],
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                    )
                else:
                    output, latency_s, prompt_tokens, completion_tokens = call_phi3_model(
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
                prompt_tokens = None
                completion_tokens = None
            ended_at = datetime.now(timezone.utc).isoformat()

            results.append(
                build_result_record(
                    sample=sample,
                    model_name=model_name,
                    model_id=active_model_id,
                    output=output,
                    error=error,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_s=latency_s,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
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
        errors = [row.get("error") for row in rows if row.get("error")]
        summary[model_name] = {
            "count": len(rows),
            "errors": len(errors),
            "first_error": errors[0] if errors else None,
            "avg_latency_s": average(
                [(row.get("metrics") or {}).get("latency_s") for row in rows]
            ),
            "avg_output_tokens": average(
                [(row.get("usage") or {}).get("completion_tokens") for row in rows]
            ),
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


def save_all_outputs(results: list[dict], output_dir: Path, include_summary: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_results(results, output_dir / "phi3_benchmark_outputs.json")

    for model_name in sorted({result["model_name"] for result in results}):
        rows = [result for result in results if result["model_name"] == model_name]
        save_results(rows, output_dir / f"{safe_filename(model_name)}_outputs.json")

    if not include_summary:
        return

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
        "--small-backend",
        choices=["auto", "torch", "onnx"],
        default="auto",
        help="Backend for Phi-3 Small. auto uses ONNX CUDA when CUDA is available.",
    )
    parser.add_argument(
        "--small-onnx-model",
        default=os.getenv("PHI3_SMALL_ONNX_MODEL", DEFAULT_SMALL_ONNX_MODEL_ID),
        help="Hugging Face repo id for the Phi-3 Small ONNX CUDA model.",
    )
    parser.add_argument(
        "--small-onnx-variant",
        choices=["cuda-int4-rtn-block-32", "cuda-fp16"],
        default=os.getenv("PHI3_SMALL_ONNX_VARIANT", DEFAULT_SMALL_ONNX_VARIANT),
        help="Subfolder variant inside the Phi-3 Small ONNX CUDA repo.",
    )
    parser.add_argument(
        "--small-onnx-dir",
        default=os.getenv("PHI3_SMALL_ONNX_DIR"),
        help="Optional local path to the ONNX model subfolder, or to a folder containing the variant subfolder.",
    )
    parser.add_argument(
        "--small-onnx-execution-provider",
        choices=["cuda", "cpu", "follow_config"],
        default=os.getenv("PHI3_SMALL_ONNX_EXECUTION_PROVIDER", DEFAULT_SMALL_ONNX_EXECUTION_PROVIDER),
        help="Execution provider for Phi-3 Small ONNX. Use cuda on Colab T4.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Force remote code for every selected model. By default only Phi-3 Small uses it.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Add automatic lexical scoring against ground_truth. By default Phi-3 matches llama4.py and only saves model outputs.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write phi3_benchmark_summary.json. By default only raw output JSON files are written.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between samples.",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Print the full model-load traceback before writing failed benchmark rows.",
    )
    return parser.parse_args()


def select_backend(model_key: str, small_backend: str, device: str) -> str:
    if model_key != "small":
        return "torch"
    if small_backend != "auto":
        return small_backend

    import torch

    return "onnx" if device != "cpu" and torch.cuda.is_available() else "torch"


def main() -> None:
    args = parse_args()
    cache_dir = normalize_hub_cache_dir(args.cache_dir)
    samples = load_samples(Path(args.data))
    if args.limit:
        samples = samples[: args.limit]

    registry = model_registry()
    selected = registry if args.model == "all" else {args.model: registry[args.model]}

    all_results = []
    for model_key, model_config in selected.items():
        backend = select_backend(model_key, args.small_backend, args.device)
        onnx_model_id = args.small_onnx_model if model_key == "small" else model_config["onnx_id"]
        results = run_model_benchmark(
            samples=samples,
            model_name=model_config["name"],
            model_id=model_config["id"],
            onnx_model_id=onnx_model_id,
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
            device=args.device,
            torch_dtype_name=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code or model_config["trust_remote_code"],
            backend=backend,
            onnx_variant=args.small_onnx_variant,
            onnx_model_dir=args.small_onnx_dir,
            onnx_execution_provider=args.small_onnx_execution_provider,
            include_scores=args.score,
            sleep_seconds=args.sleep,
            show_traceback=args.show_traceback,
        )
        all_results.extend(results)

    save_all_outputs(all_results, Path(args.output_dir), include_summary=args.summary)


if __name__ == "__main__":
    main()
