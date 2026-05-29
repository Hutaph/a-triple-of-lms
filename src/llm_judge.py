import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "bigdata_10_questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_judge"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.5"
DEFAULT_OPENROUTER_JUDGE_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 6000
DEFAULT_TEMPERATURE = 0.2

METRIC_WEIGHTS = {
    "correctness": 0.25,
    "completeness": 0.16,
    "key_point_coverage": 0.16,
    "instruction_following": 0.10,
    "reasoning_depth": 0.10,
    "production_readiness": 0.10,
    "clarity_structure": 0.07,
    "factual_grounding": 0.06,
    "code_sql_quality": 0.10,
}


class MetricScore(BaseModel):
    applicable: bool
    score: float | None
    rationale: str


class JudgeScores(BaseModel):
    correctness: MetricScore
    completeness: MetricScore
    key_point_coverage: MetricScore
    instruction_following: MetricScore
    reasoning_depth: MetricScore
    production_readiness: MetricScore
    clarity_structure: MetricScore
    factual_grounding: MetricScore
    code_sql_quality: MetricScore


class JudgeReport(BaseModel):
    overall_score: float
    scores: JudgeScores
    strengths: list[str]
    missing_or_weak_points: list[str]
    factual_or_logic_issues: list[str]
    improvement_suggestions: list[str]
    verdict: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_+#./-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sample_lookup_key(row: dict) -> str:
    sample_id = row.get("sample_id")
    if sample_id:
        return f"id::{sample_id}"

    prompt = row.get("prompt") or row.get("question")
    if prompt:
        return f"prompt::{normalize_text(prompt)[:500]}"

    raise ValueError("Cannot build lookup key without sample_id, prompt, or question.")


def load_samples(path: Path) -> dict[str, dict]:
    samples = load_json(path)
    if not isinstance(samples, list):
        raise ValueError(f"Benchmark data must be a list: {path}")

    lookup = {}
    for sample in samples:
        lookup[sample_lookup_key(sample)] = sample
    return lookup


def discover_prediction_files(root: Path) -> list[Path]:
    patterns = ["*_outputs.json", "*benchmark_results.json"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(f"**/{pattern}"))

    ignored_parts = {"llm_judge", "gpt55_judge", "judged_by_llm", "judged_by_gpt55"}
    unique_paths = []
    seen = set()
    for path in sorted(paths):
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.resolve() == DATA_FILE.resolve():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def get_model_output(row: dict) -> str:
    for field in ["model_output", "model_answer", "output", "prediction", "answer"]:
        value = row.get(field)
        if isinstance(value, str):
            return value.strip()
    return ""


def sample_for_prediction(row: dict, samples: dict[str, dict]) -> dict | None:
    keys = []
    if row.get("sample_id"):
        keys.append(f"id::{row['sample_id']}")
    if row.get("prompt"):
        keys.append(f"prompt::{normalize_text(row['prompt'])[:500]}")
    if row.get("question"):
        keys.append(f"prompt::{normalize_text(row['question'])[:500]}")

    for key in keys:
        if key in samples:
            return samples[key]
    return None


def load_prediction_rows(paths: list[Path], samples: dict[str, dict]) -> list[dict]:
    rows = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Prediction file must contain a list: {path}")

        for index, row in enumerate(data, 1):
            if not isinstance(row, dict):
                continue

            sample = sample_for_prediction(row, samples)
            if not sample:
                print(f"Skipping unmatched row {index} in {path}")
                continue

            rows.append(
                {
                    "prediction_file": str(path),
                    "prediction_index": index,
                    "sample": sample,
                    "prediction": row,
                    "candidate_output": get_model_output(row),
                }
            )
    return rows


def has_existing_result(result: dict) -> bool:
    judge = result.get("judge") or {}
    return bool(judge.get("overall_score") is not None)


def load_existing_results(path: Path) -> dict[str, dict]:
    return {result_key(row): row for row in load_existing_result_rows(path)}


def load_existing_result_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def merge_result_rows(existing_rows: list[dict], replacements: list[dict]) -> list[dict]:
    replacement_by_key = {}
    replacement_order = []
    for row in replacements:
        key = result_key(row)
        if key not in replacement_by_key:
            replacement_order.append(key)
        replacement_by_key[key] = row

    merged = []
    seen = set()
    for row in existing_rows:
        key = result_key(row)
        if key in seen:
            continue
        merged.append(replacement_by_key.get(key, row))
        seen.add(key)

    for key in replacement_order:
        if key not in seen:
            merged.append(replacement_by_key[key])
            seen.add(key)

    return merged


def normalized_path_key(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve(strict=False)
    except OSError:
        pass
    return os.path.normcase(str(path))


def result_key(row: dict) -> str:
    return "::".join(
        [
            normalized_path_key(row.get("prediction_file")),
            str(row.get("prediction_index", "")),
            str(row.get("sample_id", "")),
            str(row.get("model_name", "")),
            str(row.get("judge_model", "")),
        ]
    )


def expected_points(sample: dict) -> list[str]:
    value = sample.get("expected_key_points") or sample.get("must_have_points") or ""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        point.strip()
        for point in re.split(r";|\n", str(value))
        if point.strip()
    ]


def build_judge_input(sample: dict, prediction: dict, candidate_output: str) -> str:
    payload = {
        "task": {
            "sample_id": sample.get("sample_id") or prediction.get("id"),
            "category": sample.get("category"),
            "difficulty": sample.get("difficulty"),
            "topic": sample.get("topic"),
            "prompt": sample.get("prompt") or sample.get("question"),
            "ground_truth": sample.get("ground_truth") or sample.get("reference_answer"),
            "expected_key_points": expected_points(sample),
            "evaluation_focus": sample.get("evaluation_focus"),
            "judge_instruction": sample.get("judge_instruction"),
        },
        "candidate": {
            "model_name": prediction.get("model_name"),
            "model_id": prediction.get("model_id"),
            "output": candidate_output,
            "error": prediction.get("error"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_system_prompt() -> str:
    return (
        "You are an expert LLM-as-a-judge evaluator for Big Data, Spark, "
        "PySpark, Spark SQL, data platform architecture, and production incident analysis. "
        "Evaluate only the candidate output against the task prompt, ground truth, "
        "expected key points, and judge instruction. Use a strict 0-10 scale where "
        "10 is excellent, 7 is good but incomplete, 5 is partially correct, 3 is mostly "
        "wrong, and 0 is empty or irrelevant. Do not reward verbosity by itself. "
        "Penalize hallucinated Spark behavior, unsafe production advice, incorrect code, "
        "missing required sections, and answers that ignore explicit instructions. "
        "For code_sql_quality, mark applicable=false and score=null unless the task asks "
        "for code, SQL, or code-like implementation. Return compact JSON: each rationale "
        "must be under 18 words, each list should contain at most 3 short items, and the "
        "verdict must be one concise sentence."
    )


def judge_client(provider: str, api_key: str | None) -> Any:
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenAI SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Missing OPENROUTER_API_KEY. Put it in .env or pass --api-key.")
        return OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            api_key=key,
        )

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY. Put it in .env or pass --api-key.")

    kwargs: dict[str, str] = {"api_key": key}
    organization = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT_ID")
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    return OpenAI(**kwargs)


def extract_json_text(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text


def listify(value: Any, fallback: str | None = None) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:3]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return [fallback] if fallback else []


def metric_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def metric_aliases(metric: str) -> set[str]:
    aliases = {
        metric,
        metric.replace("_", " "),
        metric.replace("_", "-"),
    }
    aliases_by_metric = {
        "key_point_coverage": ["key points", "coverage", "expected key points"],
        "instruction_following": ["instructions", "follows instructions"],
        "reasoning_depth": ["reasoning", "depth"],
        "production_readiness": ["production", "practicality"],
        "clarity_structure": ["clarity", "structure", "presentation"],
        "factual_grounding": ["factuality", "grounding", "accuracy"],
        "code_sql_quality": ["code quality", "sql quality", "code"],
    }
    aliases.update(aliases_by_metric.get(metric, []))
    return {metric_key(alias) for alias in aliases}


def score_to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return bounded_score(value)
    except (TypeError, ValueError):
        return default


def normalize_metric_score(value: Any, default_score: float) -> dict:
    applicable = True
    score: float | None = default_score
    rationale = "Fallback from overall judge score."

    if isinstance(value, dict):
        applicable = bool(value.get("applicable", not value.get("not_applicable", False)))
        score = score_to_float(
            value.get("score", value.get("value", value.get("rating"))),
            None if not applicable else default_score,
        )
        rationale = str(
            value.get("rationale")
            or value.get("reason")
            or value.get("comment")
            or value.get("explanation")
            or rationale
        )
    elif isinstance(value, (int, float, str)):
        parsed_score = score_to_float(value)
        if parsed_score is None and isinstance(value, str) and value.strip():
            rationale = value.strip()
        else:
            score = parsed_score if parsed_score is not None else default_score

    return {
        "applicable": applicable,
        "score": score if applicable else None,
        "rationale": rationale[:220],
    }


def score_source_lookup(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        return {metric_key(str(key)): value for key, value in source.items()}
    if isinstance(source, list):
        lookup = {}
        for item in source:
            if not isinstance(item, dict):
                continue
            name = item.get("metric") or item.get("dimension") or item.get("name") or item.get("criterion")
            if name:
                lookup[metric_key(str(name))] = item
        return lookup
    return {}


def normalize_scores(data: dict, overall_score: float) -> dict:
    source = {}
    for key in ["scores", "dimension_scores", "metric_scores", "dimensions", "criteria", "rubric_scores"]:
        if key in data:
            source = score_source_lookup(data.get(key))
            if source:
                break

    scores = {}
    for metric in METRIC_WEIGHTS:
        metric_value = None
        for alias in metric_aliases(metric):
            if alias in source:
                metric_value = source[alias]
                break
        scores[metric] = normalize_metric_score(metric_value, overall_score)
    return scores


def normalize_judge_report_data(data: dict) -> JudgeReport:
    raw_overall = (
        data.get("overall_score")
        or data.get("score")
        or data.get("overall")
        or data.get("overall_rating")
        or data.get("rating")
    )
    overall_score = score_to_float(raw_overall, 0.0) or 0.0
    scores = normalize_scores(data, overall_score)

    report = {
        "overall_score": overall_score,
        "scores": scores,
        "strengths": listify(data.get("strengths") or data.get("pros") or data.get("positive_points")),
        "missing_or_weak_points": listify(
            data.get("missing_or_weak_points")
            or data.get("weaknesses")
            or data.get("missing_points")
            or data.get("areas_for_improvement")
        ),
        "factual_or_logic_issues": listify(
            data.get("factual_or_logic_issues")
            or data.get("issues")
            or data.get("errors")
            or data.get("concerns")
        ),
        "improvement_suggestions": listify(
            data.get("improvement_suggestions")
            or data.get("suggestions")
            or data.get("recommendations")
        ),
        "verdict": str(
            data.get("verdict")
            or data.get("summary")
            or data.get("rationale")
            or "Normalized from OpenRouter judge response."
        )[:300],
    }
    return JudgeReport.model_validate(report)


def parse_judge_report(content: str) -> JudgeReport:
    data = json.loads(extract_json_text(content))
    try:
        return JudgeReport.model_validate(data)
    except ValidationError:
        if not isinstance(data, dict):
            raise
        return normalize_judge_report_data(data)


def sanitize_error_message(message: str) -> str:
    message = re.sub(
        r"https://openrouter\.ai/workspaces/[^'\"\s,]+",
        "https://openrouter.ai/workspaces/[redacted]",
        message,
    )
    message = re.sub(r"sk-or-v1-[A-Za-z0-9]+", "sk-or-v1-[redacted]", message)
    message = re.sub(r"('user_id'\s*:\s*)'[^']+'", r"\1'[redacted]'", message)
    message = re.sub(r'("user_id"\s*:\s*)"[^"]+"', r'\1"[redacted]"', message)
    return message


def call_openai_responses_judge(
    client: Any,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    sample: dict,
    prediction: dict,
    candidate_output: str,
) -> tuple[JudgeReport, dict]:
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_judge_input(sample, prediction, candidate_output)},
        ],
        text_format=JudgeReport,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )
    latency_s = time.perf_counter() - started
    report = response.output_parsed
    if report is None:
        raise RuntimeError("Judge response did not include parsed structured output.")

    metadata = {
        "judge_latency_s": round(latency_s, 3),
        "judge_response_id": getattr(response, "id", None),
        "judge_created_at": datetime.now(timezone.utc).isoformat(),
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["judge_usage"] = usage.model_dump(mode="json")
        else:
            metadata["judge_usage"] = dict(usage)
    return report, metadata


def call_openrouter_chat_judge(
    client: Any,
    model: str,
    max_output_tokens: int,
    temperature: float,
    sample: dict,
    prediction: dict,
    candidate_output: str,
) -> tuple[JudgeReport, dict]:
    started = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": (
                build_system_prompt()
                + " Return only JSON using exactly these top-level keys: "
                + "overall_score, scores, strengths, missing_or_weak_points, "
                + "factual_or_logic_issues, improvement_suggestions, verdict. "
                + "scores must contain every metric key from the schema. "
                + "Do not use aliases such as score or dimension_scores. "
                + "Do not wrap the JSON in Markdown."
            ),
        },
        {"role": "user", "content": build_judge_input(sample, prediction, candidate_output)},
    ]
    schema = JudgeReport.model_json_schema()
    schema_fallback_error = None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_report",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    except Exception as schema_exc:
        schema_fallback_error = str(schema_exc)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )

    latency_s = time.perf_counter() - started
    content = response.choices[0].message.content or ""
    report = parse_judge_report(content)
    metadata = {
        "judge_latency_s": round(latency_s, 3),
        "judge_response_id": getattr(response, "id", None),
        "judge_created_at": datetime.now(timezone.utc).isoformat(),
        "judge_temperature": temperature,
    }
    if schema_fallback_error:
        metadata["structured_output_fallback_error"] = sanitize_error_message(schema_fallback_error)
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["judge_usage"] = usage.model_dump(mode="json")
        else:
            metadata["judge_usage"] = dict(usage)
    return report, metadata


def call_judge(
    client: Any,
    provider: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    temperature: float,
    sample: dict,
    prediction: dict,
    candidate_output: str,
) -> tuple[JudgeReport, dict]:
    if provider == "openrouter":
        return call_openrouter_chat_judge(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            sample=sample,
            prediction=prediction,
            candidate_output=candidate_output,
        )

    return call_openai_responses_judge(
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        sample=sample,
        prediction=prediction,
        candidate_output=candidate_output,
    )


def weighted_score(scores: dict) -> float:
    total = 0.0
    weight_total = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        value = scores.get(metric, {})
        if not value.get("applicable", True):
            continue
        score = value.get("score")
        if score is None:
            continue
        total += bounded_score(score) * weight
        weight_total += weight
    if weight_total == 0:
        return 0.0
    return round(total / weight_total, 2)


def bounded_score(value: Any) -> float:
    score = float(value)
    return min(max(score, 0.0), 10.0)


def build_result(
    item: dict,
    report: JudgeReport | None,
    judge_model: str,
    error: str | None = None,
    metadata: dict | None = None,
) -> dict:
    sample = item["sample"]
    prediction = item["prediction"]
    judge = report.model_dump(mode="json") if report else None
    metric_scores = judge["scores"] if judge else {}

    result = {
        "prediction_file": item["prediction_file"],
        "prediction_index": item["prediction_index"],
        "sample_id": sample.get("sample_id") or prediction.get("id"),
        "category": sample.get("category"),
        "difficulty": sample.get("difficulty"),
        "topic": sample.get("topic"),
        "model_name": prediction.get("model_name"),
        "model_id": prediction.get("model_id"),
        "prompt": sample.get("prompt") or sample.get("question"),
        "ground_truth": sample.get("ground_truth") or sample.get("reference_answer"),
        "expected_key_points": expected_points(sample),
        "candidate_output": item["candidate_output"],
        "candidate_error": prediction.get("error"),
        "judge_model": judge_model,
        "judge_error": error,
        "judge": judge,
        "weighted_score": weighted_score(metric_scores) if judge else None,
        "metadata": metadata or {},
    }
    return result


def average(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def summarize(results: list[dict]) -> dict:
    scored = [row for row in results if row.get("weighted_score") is not None]

    summary: dict[str, Any] = {
        "overall": {
            "count": len(results),
            "scored": len(scored),
            "judge_errors": len([row for row in results if row.get("judge_error")]),
            "avg_weighted_score": average([row.get("weighted_score") for row in scored]),
            "avg_judge_overall_score": average(
                [row.get("judge", {}).get("overall_score") for row in scored]
            ),
        },
        "by_model": {},
        "by_difficulty": {},
        "by_category": {},
        "metric_averages": {},
    }

    for metric in METRIC_WEIGHTS:
        summary["metric_averages"][metric] = average(
            [
                row.get("judge", {})
                .get("scores", {})
                .get(metric, {})
                .get("score")
                for row in scored
                if row.get("judge", {})
                .get("scores", {})
                .get(metric, {})
                .get("applicable", True)
            ]
        )

    for field, summary_key in [
        ("model_name", "by_model"),
        ("difficulty", "by_difficulty"),
        ("category", "by_category"),
    ]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in scored:
            buckets[str(row.get(field) or "unknown")].append(row)

        summary[summary_key] = {
            key: {
                "count": len(items),
                "avg_weighted_score": average([row.get("weighted_score") for row in items]),
                "avg_judge_overall_score": average(
                    [row.get("judge", {}).get("overall_score") for row in items]
                ),
            }
            for key, items in sorted(buckets.items())
        }

    return summary


def csv_row(result: dict) -> dict:
    judge = result.get("judge") or {}
    scores = judge.get("scores") or {}
    row = {
        "sample_id": result.get("sample_id"),
        "model_name": result.get("model_name"),
        "model_id": result.get("model_id"),
        "category": result.get("category"),
        "difficulty": result.get("difficulty"),
        "topic": result.get("topic"),
        "weighted_score": result.get("weighted_score"),
        "judge_overall_score": judge.get("overall_score"),
        "verdict": judge.get("verdict"),
        "judge_error": result.get("judge_error"),
        "prediction_file": result.get("prediction_file"),
    }
    for metric in METRIC_WEIGHTS:
        metric_value = scores.get(metric) or {}
        row[f"{metric}_applicable"] = metric_value.get("applicable")
        row[f"{metric}_score"] = metric_value.get("score")
        row[f"{metric}_rationale"] = metric_value.get("rationale")
    row["missing_or_weak_points"] = " | ".join(judge.get("missing_or_weak_points") or [])
    row["factual_or_logic_issues"] = " | ".join(judge.get("factual_or_logic_issues") or [])
    row["strengths"] = " | ".join(judge.get("strengths") or [])
    return row


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return

    rows = [csv_row(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge Big Data benchmark outputs with an LLM on a 0-10 metric rubric."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default="openai",
        help="API provider for the judge model.",
    )
    parser.add_argument("--data", default=str(DATA_FILE), help="Benchmark dataset JSON.")
    parser.add_argument(
        "--predictions",
        nargs="*",
        default=None,
        help="One or more output JSON files. If omitted, discovers outputs/**/*_outputs.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for judge JSON/CSV/summary outputs.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Prefer OPENAI_API_KEY or OPENROUTER_API_KEY in .env.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model id. Defaults to gpt-5.5 for OpenAI or deepseek/deepseek-chat-v3.1 for OpenRouter.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort for OpenAI Responses models.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for OpenRouter chat judge calls.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Use 0 to judge all matched rows.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing result rows when present.")
    parser.add_argument("--dry-run", action="store_true", help="Only print matched row count; do not call API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or (
        DEFAULT_OPENROUTER_JUDGE_MODEL
        if args.provider == "openrouter"
        else DEFAULT_OPENAI_JUDGE_MODEL
    )
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    tables_dir = output_dir / "tables"
    results_path = results_dir / "llm_judge_results.json"
    summary_path = results_dir / "llm_judge_summary.json"
    csv_path = tables_dir / "llm_judge_results.csv"

    samples = load_samples(data_path)
    prediction_paths = (
        [Path(path) for path in args.predictions]
        if args.predictions
        else discover_prediction_files(PROJECT_ROOT / "outputs")
    )
    if not prediction_paths:
        raise FileNotFoundError("No prediction JSON files found. Pass --predictions <file>.")

    items = load_prediction_rows(prediction_paths, samples)
    items = [item for item in items if item["candidate_output"] or item["prediction"].get("error")]
    if args.limit:
        items = items[: args.limit]

    print(f"Matched rows: {len(items)}")
    for path in prediction_paths:
        print(f"Prediction file: {path}")

    if args.dry_run:
        return

    client = judge_client(args.provider, args.api_key)
    existing_rows = load_existing_result_rows(results_path) if args.resume else []
    existing = {result_key(row): row for row in existing_rows}
    replacement_results = []

    for item in tqdm(items, desc=f"Judging with {judge_model}"):
        empty_result = build_result(
            item=item,
            report=None,
            judge_model=judge_model,
            error=None,
            metadata={},
        )
        key = result_key(empty_result)
        if key in existing and has_existing_result(existing[key]):
            replacement_results.append(existing[key])
            continue

        try:
            report, metadata = call_judge(
                client=client,
                provider=args.provider,
                model=judge_model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                sample=item["sample"],
                prediction=item["prediction"],
                candidate_output=item["candidate_output"],
            )
            result = build_result(
                item=item,
                report=report,
                judge_model=judge_model,
                metadata=metadata,
            )
        except Exception as exc:
            result = build_result(
                item=item,
                report=None,
                judge_model=judge_model,
                error=sanitize_error_message(str(exc)),
                metadata={"judge_created_at": datetime.now(timezone.utc).isoformat()},
            )

        replacement_results.append(result)
        merged_results = merge_result_rows(existing_rows, replacement_results)
        dump_json(merged_results, results_path)

        if args.sleep:
            time.sleep(args.sleep)

    results = merge_result_rows(existing_rows, replacement_results)
    summary = summarize(results)
    dump_json(results, results_path)
    dump_json(summary, summary_path)
    write_csv(results, csv_path)

    print("\nSummary")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"\nWrote: {results_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
