import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "use",
    "when",
    "where",
    "which",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_+#./-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            if left == right:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reference_keywords(reference: str, max_keywords: int = 12) -> list[str]:
    tokens = [
        token
        for token in tokenize(reference)
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    ]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(max_keywords)]


def keyword_coverage(prediction: str, reference: str) -> float:
    keywords = reference_keywords(reference)
    if not keywords:
        return 0.0
    pred_tokens = set(tokenize(prediction))
    return sum(1 for keyword in keywords if keyword in pred_tokens) / len(keywords)


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    ]


def must_have_point_coverage(prediction: str, must_have_points: list[str] | None) -> float:
    if not must_have_points:
        return 0.0

    pred_tokens = set(content_tokens(prediction))
    if not pred_tokens:
        return 0.0

    point_scores = []
    for point in must_have_points:
        point_tokens = set(content_tokens(point))
        if not point_tokens:
            continue
        recall = len(point_tokens & pred_tokens) / len(point_tokens)
        point_scores.append(min(recall / 0.6, 1.0))

    return average(point_scores)


def instruction_following_score(prediction: str) -> float:
    tokens = tokenize(prediction)
    if not tokens:
        return 0.0

    word_count = len(tokens)
    if 40 <= word_count <= 220:
        length_score = 1.0
    elif 25 <= word_count < 40 or 220 < word_count <= 280:
        length_score = 0.7
    else:
        length_score = 0.35

    normalized = normalize_text(prediction)
    refusal_patterns = [
        "i cannot",
        "i can t",
        "as an ai",
        "cannot answer",
        "not enough information",
    ]
    no_refusal_score = 0.0 if any(pattern in normalized for pattern in refusal_patterns) else 1.0

    spark_terms = {
        "spark",
        "shuffle",
        "partition",
        "executor",
        "driver",
        "join",
        "dataframe",
        "streaming",
        "parquet",
        "delta",
        "cache",
        "memory",
        "stage",
        "task",
    }
    relevance_score = 1.0 if spark_terms & set(tokens) else 0.6

    tradeoff_terms = {
        "trade-off",
        "tradeoff",
        "cost",
        "latency",
        "memory",
        "shuffle",
        "skew",
        "production",
        "monitor",
        "failure",
        "bottleneck",
        "depends",
    }
    tradeoff_score = 1.0 if tradeoff_terms & set(tokens) else 0.5

    return average([length_score, no_refusal_score, relevance_score, tradeoff_score])


def composite_score(
    prediction: str,
    reference: str,
    must_have_points: list[str] | None = None,
) -> dict[str, float]:
    f1 = token_f1(prediction, reference)
    rouge = rouge_l(prediction, reference)
    coverage = keyword_coverage(prediction, reference)
    point_coverage = must_have_point_coverage(prediction, must_have_points)
    instruction_score = instruction_following_score(prediction)
    score = (
        0.25 * f1
        + 0.20 * rouge
        + 0.25 * coverage
        + 0.20 * point_coverage
        + 0.10 * instruction_score
    )
    return {
        "score": round(score, 4),
        "token_f1": round(f1, 4),
        "rouge_l": round(rouge, 4),
        "keyword_coverage": round(coverage, 4),
        "must_have_point_coverage": round(point_coverage, 4),
        "instruction_following": round(instruction_score, 4),
    }


def build_prompt(question: str) -> str:
    return (
        "Answer this Spark or Big Data interview question like a senior data engineer. "
        "Be concise but include trade-offs and production considerations when relevant.\n\n"
        f"Question: {question}"
    )


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> tuple[str, float, int]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, latency, len(generated_ids)


def average(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[dict]) -> dict:
    overall = {
        "count": len(results),
        "avg_score": round(average([r["score"] for r in results]), 4),
        "avg_token_f1": round(average([r["token_f1"] for r in results]), 4),
        "avg_rouge_l": round(average([r["rouge_l"] for r in results]), 4),
        "avg_keyword_coverage": round(average([r["keyword_coverage"] for r in results]), 4),
        "avg_must_have_point_coverage": round(
            average([r["must_have_point_coverage"] for r in results]), 4
        ),
        "avg_instruction_following": round(
            average([r["instruction_following"] for r in results]), 4
        ),
        "avg_latency_s": round(average([r["latency_s"] for r in results]), 3),
        "avg_output_tokens": round(average([r["output_tokens"] for r in results]), 1),
    }

    groups = {}
    for field in ["difficulty", "category"]:
        bucket = defaultdict(list)
        for result in results:
            bucket[result[field]].append(result)
        groups[field] = {
            key: {
                "count": len(items),
                "avg_score": round(average([r["score"] for r in items]), 4),
                "avg_must_have_point_coverage": round(
                    average([r["must_have_point_coverage"] for r in items]), 4
                ),
                "avg_instruction_following": round(
                    average([r["instruction_following"] for r in items]), 4
                ),
                "avg_latency_s": round(average([r["latency_s"] for r in items]), 3),
            }
            for key, items in sorted(bucket.items())
        }

    return {"overall": overall, "by": groups}


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def write_human_review_template(results: list[dict], path: Path) -> None:
    fields = [
        "id",
        "question",
        "category",
        "difficulty",
        "model_answer",
        "must_have_points",
        "auto_score",
        "correctness_1_5",
        "completeness_1_5",
        "tradeoff_1_5",
        "production_relevance_1_5",
        "clarity_1_5",
        "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "question": result["question"],
                    "category": result["category"],
                    "difficulty": result["difficulty"],
                    "model_answer": result["model_answer"],
                    "must_have_points": json.dumps(
                        result.get("must_have_points", []),
                        ensure_ascii=False,
                    ),
                    "auto_score": result["score"],
                    "correctness_1_5": "",
                    "completeness_1_5": "",
                    "tradeoff_1_5": "",
                    "production_relevance_1_5": "",
                    "clarity_1_5": "",
                    "human_notes": "",
                }
            )


def write_low_score_report(results: list[dict], path: Path, bottom_n: int = 10) -> None:
    lowest = sorted(results, key=lambda row: row["score"])[:bottom_n]
    lines = [
        "# Low Score Analysis",
        "",
        "Các câu dưới đây có điểm tự động thấp nhất. Nên đọc thủ công để phân biệt lỗi thật của model với hạn chế của metric token/keyword.",
        "",
    ]
    for result in lowest:
        lines.extend(
            [
                f"## {result['id']}. {result['question']}",
                "",
                f"- Category: `{result['category']}`",
                f"- Difficulty: `{result['difficulty']}`",
                f"- Auto score: `{result['score']}`",
                f"- Token F1: `{result['token_f1']}`",
                f"- ROUGE-L: `{result['rouge_l']}`",
                f"- Keyword coverage: `{result['keyword_coverage']}`",
                f"- Must-have point coverage: `{result['must_have_point_coverage']}`",
                f"- Instruction following: `{result['instruction_following']}`",
                "",
                "Must-have points:",
            ]
        )
        for point in result.get("must_have_points", []):
            lines.append(f"- {point}")
        lines.extend(
            [
                "",
                "Model answer:",
                "",
                result["model_answer"],
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_visualizations(results: list[dict], summary: dict, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping visualizations.")
        return []

    if not results:
        return []

    chart_paths = []

    difficulty_items = summary["by"]["difficulty"]
    raw_labels = list(difficulty_items.keys())
    labels = [
        f"{label}\n(n={difficulty_items[label]['count']})"
        for label in raw_labels
    ]
    scores = [difficulty_items[label]["avg_score"] for label in raw_labels]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, scores, color="#4f7cff")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.015,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0, 1)
    plt.title("Average Score by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "score_by_difficulty.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_paths.append(str(path))

    category_items = summary["by"]["category"]
    category_scores = sorted(
        [(key, value["avg_score"], value["count"]) for key, value in category_items.items()],
        key=lambda item: item[1],
    )
    labels = [f"{item[0]} (n={item[2]})" for item in category_scores]
    scores = [item[1] for item in category_scores]
    plt.figure(figsize=(9, max(4, 0.35 * len(labels))))
    bars = plt.barh(labels, scores, color="#16a085")
    for bar, score in zip(bars, scores):
        plt.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=8,
        )
    plt.xlim(0, 1)
    plt.title("Average Score by Category")
    plt.xlabel("Score")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "score_by_category.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_paths.append(str(path))

    raw_labels = list(difficulty_items.keys())
    labels = [
        f"{label}\n(n={difficulty_items[label]['count']})"
        for label in raw_labels
    ]
    latencies = [difficulty_items[label]["avg_latency_s"] for label in raw_labels]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, latencies, color="#f39c12")
    for bar, latency in zip(bars, latencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            latency + 0.05,
            f"{latency:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Average Latency by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Seconds")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "latency_by_difficulty.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_paths.append(str(path))

    plt.figure(figsize=(7, 5))
    difficulty_colors = {"easy": "#2ecc71", "medium": "#f1c40f", "hard": "#e74c3c"}
    for difficulty in sorted({result["difficulty"] for result in results}):
        subset = [result for result in results if result["difficulty"] == difficulty]
        plt.scatter(
            [result["latency_s"] for result in subset],
            [result["score"] for result in subset],
            label=difficulty,
            alpha=0.8,
            color=difficulty_colors.get(difficulty, "#4f7cff"),
        )
    plt.ylim(0, 1)
    plt.title("Score vs Latency")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Score")
    plt.grid(alpha=0.25)
    plt.legend(title="Difficulty")
    plt.tight_layout()
    path = output_dir / "score_vs_latency.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_paths.append(str(path))

    return chart_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local LLM on Spark interview QA data.")
    parser.add_argument("--data", default="data/spark_interview_questions.json")
    parser.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for the full dataset.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    tokenizer, model = load_model(args.model_id, args.device)

    results = []
    for index, item in enumerate(dataset, 1):
        prompt = build_prompt(item["question"])
        model_answer, latency, output_tokens = generate_answer(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        scores = composite_score(
            model_answer,
            item["reference_answer"],
            item.get("must_have_points"),
        )
        result = {
            "id": index,
            "question": item["question"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "reference_answer": item["reference_answer"],
            "must_have_points": item.get("must_have_points", []),
            "model_answer": model_answer,
            "latency_s": round(latency, 3),
            "output_tokens": output_tokens,
            **scores,
        }
        results.append(result)
        print(
            f"[{index:03d}/{len(dataset):03d}] "
            f"{item['difficulty']} {item['category']} "
            f"score={result['score']:.4f} latency={result['latency_s']:.1f}s"
        )

    summary = summarize(results)
    details_path = output_dir / "spark_qa_benchmark_results.json"
    summary_path = output_dir / "spark_qa_benchmark_summary.json"
    csv_path = output_dir / "spark_qa_benchmark_results.csv"
    human_review_path = output_dir / "human_review_template.csv"
    low_score_report_path = output_dir / "low_score_analysis.md"

    details_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(results, csv_path)
    write_human_review_template(results, human_review_path)
    write_low_score_report(results, low_score_report_path)
    chart_paths = save_visualizations(results, summary, output_dir)

    print("\nSummary")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"\nWrote: {details_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {human_review_path}")
    print(f"Wrote: {low_score_report_path}")
    for chart_path in chart_paths:
        print(f"Wrote: {chart_path}")


if __name__ == "__main__":
    main()
