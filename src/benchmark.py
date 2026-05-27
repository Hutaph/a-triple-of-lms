import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

BENCHMARK_NAME = "General Big Data LM Benchmark"
BENCHMARK_VERSION = "v1.0"
SCORE_SCALE = "0-10"

DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 2


RUBRIC = {
    "correctness": "Technical correctness of the answer.",
    "completeness": "Coverage of required points and expected key concepts.",
    "reasoning_quality": "Logical structure, depth, and explanation quality.",
    "big_data_relevance": "Relevance to Spark, Big Data, and production data engineering.",
    "practicality": "Usefulness in real engineering scenarios.",
    "code_quality": (
        "Correctness, idiomatic style, scalability, and maintainability of code. "
        "Use null for non-code tasks."
    ),
    "instruction_following": "How well the answer follows the prompt constraints.",
}


LIMITATIONS = [
    "Each sample was evaluated once, so output variance across repeated runs was not measured.",
    "Latency depends on API provider routing, network condition, provider load, and output length.",
    "LLM-as-a-judge evaluation may contain subjective bias despite using a fixed rubric.",
    "The benchmark focuses on Big Data and Spark tasks, not general-purpose reasoning.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def average(values: list[float | int | None]) -> float | None:
    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return None
    return round(mean(clean_values), 3)


def median_value(values: list[float | int | None]) -> float | None:
    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return None
    return round(median(clean_values), 3)


def load_environment() -> tuple[OpenAI, dict[str, str]]:
    load_dotenv(PROJECT_ROOT / ".env")

    judge_api_key = os.getenv("OPENROUTER_API_KEY")
    judge_base_url = os.getenv("JUDGE_BASE_URL")
    judge_model_id = os.getenv("JUDGE_MODEL_ID")
    judge_model_name = os.getenv("JUDGE_MODEL_NAME")
    judge_provider = os.getenv("JUDGE_PROVIDER")

    if not judge_api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Please add it to your .env file."
        )

    client = OpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    judge_config = {
        "model_name": judge_model_name,
        "model_id": judge_model_id,
        "provider": judge_provider,
        "base_url": judge_base_url,
    }

    return client, judge_config


def load_dataset(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Dataset JSON must be a list of samples.")

    dataset_map = {}

    for sample in samples:
        sample_id = sample.get("sample_id")
        if not sample_id:
            raise ValueError("Every sample must have a sample_id.")
        dataset_map[sample_id] = sample

    return dataset_map


def discover_output_files(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    json_files = []

    for path in outputs_dir.rglob("*.json"):
        if "benchmark" in path.parts:
            continue
        json_files.append(path)

    return sorted(json_files)


def load_output_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Output file must be a list of records: {path}")

    return records


def infer_model_key(output_file: Path) -> str:
    """
    examples:
    outputs/llama4/llama4_scout_outputs.json -> llama4_scout
    outputs/qwen3/qwen3_outputs.json -> qwen3
    """
    stem = output_file.stem

    if stem.endswith("_outputs"):
        return stem.removesuffix("_outputs")

    if stem.endswith("_output"):
        return stem.removesuffix("_output")

    return stem


def extract_json_object(text: str) -> dict:
    """
    The judge should return strict JSON. This fallback extracts the first JSON object
    if the model accidentally wraps it with prose or markdown.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge response is not valid JSON:\n{text}")

    return json.loads(match.group(0))


def is_code_task(category: str | None) -> bool:
    return (category or "").strip().lower() == "code generation"


def build_judge_prompt(
    dataset_sample: dict,
    output_record: dict,
) -> str:
    category = dataset_sample.get("category")
    code_quality_rule = (
        "For code_quality, provide a 0-10 score because this is a code generation task."
        if is_code_task(category)
        else "For code_quality, return null because this is not a code generation task."
    )

    return f"""
You are an expert evaluator for Big Data, Spark, and Data Engineering benchmark answers.

Evaluate the candidate model output against:
1. The original prompt
2. The ground truth answer
3. The expected key points
4. The evaluation focus
5. The judge instruction

Use a 0-10 score scale:
- 0-2: severely wrong, irrelevant, or unusable
- 3-4: partially correct but missing major ideas
- 5-6: acceptable but incomplete or shallow
- 7-8: good, mostly correct, minor omissions
- 9: excellent, almost complete
- 10: perfect or near-perfect

Important:
- Do not reward verbosity alone.
- Penalize hallucinated technical claims.
- Penalize code that looks plausible but is logically wrong.
- For Big Data/Spark tasks, prioritize correctness, scalability, production relevance, and practical trade-offs.
- {code_quality_rule}

Return ONLY valid JSON with this exact schema:
{{
  "scores": {{
    "correctness": number,
    "completeness": number,
    "reasoning_quality": number,
    "big_data_relevance": number,
    "practicality": number,
    "code_quality": number or null,
    "instruction_following": number
  }},
  "quality_score": number,
  "strengths": [string],
  "weaknesses": [string],
  "main_errors": [string],
  "short_comment": string
}}

Original prompt:
{dataset_sample.get("prompt", "")}

Ground truth:
{dataset_sample.get("ground_truth", "")}

Expected key points:
{dataset_sample.get("expected_key_points", "")}

Evaluation focus:
{dataset_sample.get("evaluation_focus", "")}

Judge instruction:
{dataset_sample.get("judge_instruction", "")}

Candidate model:
{output_record.get("model_name")} ({output_record.get("model_id")})

Candidate model output:
{output_record.get("model_output", "")}
""".strip()


def call_judge_model(
    client: OpenAI,
    judge_config: dict[str, str],
    judge_prompt: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_config["model_id"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict benchmark evaluator. "
                            "Return only valid JSON. Do not include markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": judge_prompt,
                    },
                ],
                temperature=0,
                max_tokens=900,
            )

            content = response.choices[0].message.content
            return extract_json_object(content)

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Judge model failed after retries: {last_error}")


def normalize_scores(evaluation: dict, category: str | None) -> dict:
    scores = evaluation.get("scores", {})

    normalized = {
        "correctness": safe_float(scores.get("correctness")),
        "completeness": safe_float(scores.get("completeness")),
        "reasoning_quality": safe_float(scores.get("reasoning_quality")),
        "big_data_relevance": safe_float(scores.get("big_data_relevance")),
        "practicality": safe_float(scores.get("practicality")),
        "code_quality": safe_float(scores.get("code_quality")),
        "instruction_following": safe_float(scores.get("instruction_following")),
    }

    if not is_code_task(category):
        normalized["code_quality"] = None

    for key, value in normalized.items():
        if value is None:
            continue
        normalized[key] = max(0.0, min(10.0, round(value, 3)))

    return normalized


def compute_quality_score(scores: dict) -> float | None:
    return average([
        scores.get("correctness"),
        scores.get("completeness"),
        scores.get("reasoning_quality"),
        scores.get("big_data_relevance"),
        scores.get("practicality"),
        scores.get("code_quality"),
        scores.get("instruction_following"),
    ])


def build_raw_output_metrics(output_record: dict) -> dict:
    usage = output_record.get("usage") or {}
    metrics = output_record.get("metrics") or {}

    return {
        "latency_s": safe_float(metrics.get("latency_s")),
        "tokens_per_second": safe_float(metrics.get("tokens_per_second")),
        "prompt_tokens": safe_int(usage.get("prompt_tokens")),
        "completion_tokens": safe_int(usage.get("completion_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
        "output_char_count": safe_int(metrics.get("output_char_count")),
        "output_word_count": safe_int(metrics.get("output_word_count")),
    }


def build_sample_evaluation(
    dataset_sample: dict,
    output_record: dict,
    judge_config: dict[str, str],
    judge_result: dict,
) -> dict:
    category = dataset_sample.get("category")
    scores = normalize_scores(judge_result, category)
    quality_score = compute_quality_score(scores)

    metadata = output_record.get("metadata") or {}

    return {
        "sample_id": output_record.get("sample_id"),
        "category": output_record.get("category") or dataset_sample.get("category"),
        "difficulty": output_record.get("difficulty") or dataset_sample.get("difficulty"),
        "topic": output_record.get("topic") or dataset_sample.get("topic"),

        "prompt": dataset_sample.get("prompt"),
        "ground_truth": dataset_sample.get("ground_truth"),
        "expected_key_points": dataset_sample.get("expected_key_points"),
        "model_output": output_record.get("model_output"),

        "candidate_model": {
            "model_name": output_record.get("model_name"),
            "model_id": output_record.get("model_id"),
        },

        "raw_output_metrics": build_raw_output_metrics(output_record),

        "scores": scores,
        "quality_score": quality_score,

        "judge_feedback": {
            "strengths": judge_result.get("strengths", []),
            "weaknesses": judge_result.get("weaknesses", []),
            "main_errors": judge_result.get("main_errors", []),
            "short_comment": judge_result.get("short_comment", ""),
        },

        "metadata": {
            "candidate_started_at": metadata.get("started_at"),
            "candidate_ended_at": metadata.get("ended_at"),
            "judged_at": utc_now(),
            "judge_model_name": judge_config["model_name"],
            "judge_model_id": judge_config["model_id"],
        },
    }


def summarize_by_group(
    sample_evaluations: list[dict],
    group_key: str,
) -> dict:
    grouped = {}

    for item in sample_evaluations:
        group_value = item.get(group_key) or "unknown"
        grouped.setdefault(group_value, []).append(item)

    summary = {}

    for group_value, items in grouped.items():
        summary[group_value] = {
            "num_samples": len(items),
            "average_quality_score": average([
                item.get("quality_score") for item in items
            ]),
            "average_latency_s": average([
                item.get("raw_output_metrics", {}).get("latency_s")
                for item in items
            ]),
        }

    return summary


def build_summary(sample_evaluations: list[dict]) -> dict:
    scores = [item.get("scores", {}) for item in sample_evaluations]
    raw_metrics = [item.get("raw_output_metrics", {}) for item in sample_evaluations]

    return {
        "average_quality_score": average([
            item.get("quality_score") for item in sample_evaluations
        ]),
        "average_correctness": average([
            score.get("correctness") for score in scores
        ]),
        "average_completeness": average([
            score.get("completeness") for score in scores
        ]),
        "average_reasoning_quality": average([
            score.get("reasoning_quality") for score in scores
        ]),
        "average_big_data_relevance": average([
            score.get("big_data_relevance") for score in scores
        ]),
        "average_practicality": average([
            score.get("practicality") for score in scores
        ]),
        "average_code_quality": average([
            score.get("code_quality") for score in scores
        ]),
        "average_instruction_following": average([
            score.get("instruction_following") for score in scores
        ]),

        "average_latency_s": average([
            metric.get("latency_s") for metric in raw_metrics
        ]),
        "median_latency_s": median_value([
            metric.get("latency_s") for metric in raw_metrics
        ]),
        "average_tokens_per_second": average([
            metric.get("tokens_per_second") for metric in raw_metrics
        ]),
        "total_prompt_tokens": sum(
            metric.get("prompt_tokens") or 0 for metric in raw_metrics
        ),
        "total_completion_tokens": sum(
            metric.get("completion_tokens") or 0 for metric in raw_metrics
        ),
        "total_tokens": sum(
            metric.get("total_tokens") or 0 for metric in raw_metrics
        ),

        "average_by_category": summarize_by_group(sample_evaluations, "category"),
        "average_by_difficulty": summarize_by_group(sample_evaluations, "difficulty"),
    }


def build_generation_params_summary(records: list[dict]) -> dict:
    temperatures = sorted({
        (record.get("generation_params") or {}).get("temperature")
        for record in records
        if (record.get("generation_params") or {}).get("temperature") is not None
    })

    max_tokens_values = sorted({
        (record.get("generation_params") or {}).get("max_tokens")
        for record in records
        if (record.get("generation_params") or {}).get("max_tokens") is not None
    })

    return {
        "temperatures_used": temperatures,
        "max_tokens_values_used": max_tokens_values,
        "max_tokens_policy": "Per-sample max_tokens from benchmark dataset",
    }


def build_benchmark_report(
    source_output_file: Path,
    output_records: list[dict],
    sample_evaluations: list[dict],
    judge_config: dict[str, str],
    dataset_file: Path,
) -> dict:
    first_record = output_records[0] if output_records else {}

    num_success = sum(1 for item in output_records if item.get("status") == "success")
    num_failed = len(output_records) - num_success

    return {
        "benchmark_metadata": {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_version": BENCHMARK_VERSION,
            "created_at": utc_now(),
            "score_scale": SCORE_SCALE,
            "source_output_file": str(source_output_file.relative_to(PROJECT_ROOT)),
            "dataset_file": str(dataset_file.relative_to(PROJECT_ROOT)),
            "num_samples": len(output_records),
            "num_success": num_success,
            "num_failed": num_failed,
        },

        "candidate_model": {
            "model_name": first_record.get("model_name"),
            "model_id": first_record.get("model_id"),
            "provider": "OpenRouter",
            "system_prompt": first_record.get("system_prompt"),
            "generation_params_summary": build_generation_params_summary(output_records),
        },

        "judge_model": {
            "model_name": judge_config["model_name"],
            "model_id": judge_config["model_id"],
            "provider": judge_config["provider"],
            "score_scale": SCORE_SCALE,
            "rubric": RUBRIC,
        },

        "sample_evaluations": sample_evaluations,
        "summary": build_summary(sample_evaluations),
        "limitations": LIMITATIONS,
    }


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved benchmark result: {output_path}")


def benchmark_one_output_file(
    client: OpenAI,
    judge_config: dict[str, str],
    dataset_map: dict[str, dict],
    dataset_file: Path,
    output_file: Path,
    results_dir: Path,
    sleep_seconds: float,
    max_retries: int,
) -> Path:
    output_records = load_output_records(output_file)
    model_key = infer_model_key(output_file)

    sample_evaluations = []

    print(f"\nBenchmarking output file: {output_file}")

    for record in tqdm(output_records, desc=f"Judging {model_key}"):
        sample_id = record.get("sample_id")

        if sample_id not in dataset_map:
            raise KeyError(
                f"sample_id={sample_id} from {output_file} not found in dataset."
            )

        dataset_sample = dataset_map[sample_id]

        if record.get("status") != "success" or record.get("error"):
            scores = {
                "correctness": 0,
                "completeness": 0,
                "reasoning_quality": 0,
                "big_data_relevance": 0,
                "practicality": 0,
                "code_quality": 0 if is_code_task(dataset_sample.get("category")) else None,
                "instruction_following": 0,
            }

            judge_result = {
                "scores": scores,
                "quality_score": 0,
                "strengths": [],
                "weaknesses": ["Candidate model request failed."],
                "main_errors": [record.get("error") or "Unknown error"],
                "short_comment": "No valid model output to evaluate.",
            }
        else:
            judge_prompt = build_judge_prompt(dataset_sample, record)
            judge_result = call_judge_model(
                client=client,
                judge_config=judge_config,
                judge_prompt=judge_prompt,
                max_retries=max_retries,
            )

        sample_eval = build_sample_evaluation(
            dataset_sample=dataset_sample,
            output_record=record,
            judge_config=judge_config,
            judge_result=judge_result,
        )

        sample_evaluations.append(sample_eval)
        time.sleep(sleep_seconds)

    benchmark_report = build_benchmark_report(
        source_output_file=output_file,
        output_records=output_records,
        sample_evaluations=sample_evaluations,
        judge_config=judge_config,
        dataset_file=dataset_file,
    )

    output_path = results_dir / f"{model_key}_benchmark.json"
    save_json(benchmark_report, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all raw LM output JSON files under outputs/ and save benchmark results."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Directory containing raw model output JSON files.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save benchmark result JSON files.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATA_FILE),
        help="Benchmark dataset JSON file.",
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional: evaluate only one raw output JSON file.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between judge model calls.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries for judge model calls.",
    )

    args = parser.parse_args()

    dataset_file = Path(args.dataset).resolve()

    input_dir = Path(args.input_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    client, judge_config = load_environment()
    dataset_map = load_dataset(dataset_file)

    if args.file:
        output_files = [Path(args.file).resolve()]
    else:
        output_files = discover_output_files(input_dir)

    if not output_files:
        raise FileNotFoundError(f"No JSON output files found under: {input_dir}")

    print("Found output files:")
    for file in output_files:
        print(f"- {file}")

    saved_files = []

    for output_file in output_files:
        saved_path = benchmark_one_output_file(
            client=client,
            judge_config=judge_config,
            dataset_map=dataset_map,
            dataset_file=dataset_file,
            output_file=output_file,
            results_dir=results_dir,
            sleep_seconds=args.sleep,
            max_retries=args.max_retries,
        )
        saved_files.append(saved_path)

    print("\nDone. Benchmark files:")
    for saved_file in saved_files:
        print(f"- {saved_file}")


if __name__ == "__main__":
    main()