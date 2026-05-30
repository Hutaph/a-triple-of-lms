from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
JUDGE_RESULTS_PATH = OUTPUTS_DIR / "llm_judge" / "results" / "llm_judge_results.json"
SLIDE_DIR = PROJECT_ROOT / "slide"

TASK_FOLDERS = {
    "Theory Q&A": "theory_qa",
    "Code Generation": "code_generation",
    "Incident Analysis": "incident_analysis",
}

MODEL_LABELS = {
    "Qwen3-235B-A22B-Thinking": "Qwen3 235B-A22B Thinking",
    "Qwen3-32B": "Qwen3-32B Thinking",
    "Qwen3-14B": "Qwen3-14B Thinking",
    "Qwen3-4B-Thinking-HF": "Qwen3 4B Thinking",
}

CHART_COLORS = {
    "score": "#4C78A8",
    "latency": "#D28E5D",
    "tokens": "#5B9A8B",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_raw_metrics() -> dict[tuple[str, str], dict[str, float]]:
    metrics_by_key = {}
    for path in sorted(OUTPUTS_DIR.rglob("*_outputs.json")):
        if "llm_judge" in path.parts:
            continue

        rows = load_json(path)
        if not isinstance(rows, list):
            continue

        for row in rows:
            model_name = row.get("model_name")
            sample_id = row.get("sample_id")
            if not model_name or not sample_id:
                continue

            latency_s = row.get("latency_s")
            if latency_s is None:
                latency_s = (row.get("metrics") or {}).get("latency_s")
            completion_tokens = (row.get("usage") or {}).get("completion_tokens")

            if latency_s is None or completion_tokens is None:
                raise ValueError(f"Missing raw metrics for {model_name} / {sample_id} in {path}")

            key = (model_name, sample_id)
            if key in metrics_by_key:
                raise ValueError(f"Duplicate raw output metrics for {model_name} / {sample_id}")
            metrics_by_key[key] = {
                "latency_s": float(latency_s),
                "completion_tokens": float(completion_tokens),
            }

    return metrics_by_key


def build_task_metrics() -> dict[str, dict[str, dict[str, float]]]:
    judge_rows = load_json(JUDGE_RESULTS_PATH)
    raw_metrics = load_raw_metrics()
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in judge_rows:
        category = row.get("category")
        model_name = row.get("model_name")
        sample_id = row.get("sample_id")
        if category not in TASK_FOLDERS:
            continue

        raw = raw_metrics[(model_name, sample_id)]
        grouped[category][model_name]["score"].append(float(row["weighted_score"]))
        grouped[category][model_name]["latency_s"].append(raw["latency_s"])
        grouped[category][model_name]["completion_tokens"].append(raw["completion_tokens"])

    aggregated = {}
    for category, models in grouped.items():
        aggregated[category] = {}
        for model_name, values in models.items():
            aggregated[category][model_name] = {
                metric: mean(metric_values) for metric, metric_values in values.items()
            }
    return aggregated


def display_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def render_bar_chart(
    *,
    values_by_model: dict[str, float],
    output_path: Path,
    title: str,
    x_label: str,
    color: str,
    higher_is_better: bool,
    number_format: str,
    footer_note: str | None = None,
) -> None:
    ordered = sorted(values_by_model.items(), key=lambda item: item[1], reverse=higher_is_better)
    labels = [display_label(model_name) for model_name, _ in ordered]
    values = [value for _, value in ordered]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(labels, values, color=color, height=0.62)
    ax.invert_yaxis()

    max_value = max(values)
    ax.set_xlim(0, max_value * 1.17)
    ax.set_xlabel(x_label, color="#404040", labelpad=10)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D8D8D4", linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#5C5C5C")
    ax.tick_params(axis="y", colors="#303030", labelsize=11, length=0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max_value * 0.018,
            bar.get_y() + bar.get_height() / 2,
            number_format.format(value),
            va="center",
            ha="left",
            fontsize=11,
            color="#303030",
            fontweight="semibold",
        )

    fig.suptitle(title, x=0.5, y=0.94, ha="center", fontsize=20, fontweight="semibold", color="#202020")
    if footer_note:
        fig.text(0.5, 0.035, footer_note, ha="center", fontsize=8.5, color="#666666")
    fig.subplots_adjust(left=0.28, right=0.94, top=0.84, bottom=0.16 if footer_note else 0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_quality_latency_tradeoff(
    *,
    values_by_model: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    family_colors = {
        "qwen3": "#4C78A8",
        "llama4": "#D28E5D",
        "phi3": "#5B9A8B",
    }

    def family(model_name: str) -> str:
        if model_name.startswith("Qwen"):
            return "qwen3"
        if model_name.startswith("Llama"):
            return "llama4"
        return "phi3"

    max_latency = max(values["latency_s"] for values in values_by_model.values())
    label_offsets = {
        "Qwen3-4B-Thinking-HF": (7, -14),
        "Qwen3-32B": (7, 8),
        "Qwen3-14B": (7, 8),
    }
    for model_name, values in values_by_model.items():
        latency = values["latency_s"]
        score = values["score"]
        color = family_colors[family(model_name)]
        ax.scatter(latency, score, s=145, color=color, edgecolors="#FFFFFF", linewidth=1.3, zorder=3)
        ax.annotate(
            display_label(model_name),
            (latency, score),
            xytext=label_offsets.get(model_name, (7, 5)),
            textcoords="offset points",
            fontsize=9.5,
            color="#303030",
        )

    ax.set_xlim(left=0, right=max_latency * 1.18)
    ax.set_ylim(4, 10)
    ax.set_xlabel("Average response time (seconds)", color="#404040", labelpad=10)
    ax.set_ylabel("Average weighted score / 10", color="#404040", labelpad=10)
    ax.grid(color="#D8D8D4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#5C5C5C")

    fig.suptitle(
        "Quality vs. Response Time Trade-off - 10 Questions",
        x=0.5,
        y=0.94,
        ha="center",
        fontsize=20,
        fontweight="semibold",
        color="#202020",
    )
    fig.text(
        0.5,
        0.035,
        "Upper-left is better. Phi-3 runs locally; most other models use API providers, so latency is observational.",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    fig.subplots_adjust(left=0.12, right=0.88, top=0.84, bottom=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_tradeoff_scatter(
    *,
    values_by_model: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    footer_note: str,
    x_upper_factor: float = 1.18,
    y_lower: float | None = None,
    y_upper_factor: float = 1.15,
    label_offsets: dict[str, tuple[int, int]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    family_colors = {
        "qwen3": "#4C78A8",
        "llama4": "#D28E5D",
        "phi3": "#5B9A8B",
    }

    def family(model_name: str) -> str:
        if model_name.startswith("Qwen"):
            return "qwen3"
        if model_name.startswith("Llama"):
            return "llama4"
        return "phi3"

    x_values = [values[x_metric] for values in values_by_model.values()]
    y_values = [values[y_metric] for values in values_by_model.values()]
    label_offsets = label_offsets or {}
    for model_name, values in values_by_model.items():
        x_value = values[x_metric]
        y_value = values[y_metric]
        color = family_colors[family(model_name)]
        ax.scatter(x_value, y_value, s=145, color=color, edgecolors="#FFFFFF", linewidth=1.3, zorder=3)
        ax.annotate(
            display_label(model_name),
            (x_value, y_value),
            xytext=label_offsets.get(model_name, (7, 5)),
            textcoords="offset points",
            fontsize=9.5,
            color="#303030",
        )

    ax.set_xlim(left=0, right=max(x_values) * x_upper_factor)
    if y_lower is not None:
        ax.set_ylim(bottom=y_lower, top=max(y_values) * y_upper_factor)
    else:
        ax.set_ylim(bottom=0, top=max(y_values) * y_upper_factor)
    ax.set_xlabel(x_label, color="#404040", labelpad=10)
    ax.set_ylabel(y_label, color="#404040", labelpad=10)
    ax.grid(color="#D8D8D4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#5C5C5C")

    fig.suptitle(
        title,
        x=0.5,
        y=0.94,
        ha="center",
        fontsize=20,
        fontweight="semibold",
        color="#202020",
    )
    fig.text(0.5, 0.035, footer_note, ha="center", fontsize=8.5, color="#666666")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.84, bottom=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    task_metrics = build_task_metrics()
    judge_rows = load_json(JUDGE_RESULTS_PATH)
    chart_specs = [
        {
            "metric": "score",
            "filename": "01_model_comparison.png",
            "title": "Model Quality Comparison",
            "x_label": "Average weighted score / 10",
            "color": CHART_COLORS["score"],
            "higher_is_better": True,
            "number_format": "{:.2f}",
        },
        {
            "metric": "latency_s",
            "filename": "02_response_time_comparison.png",
            "title": "Response Time Comparison",
            "x_label": "Average response time (seconds)",
            "color": CHART_COLORS["latency"],
            "higher_is_better": False,
            "number_format": "{:.1f}s",
        },
        {
            "metric": "completion_tokens",
            "filename": "03_generated_tokens_comparison.png",
            "title": "Generated Tokens Comparison",
            "x_label": "Average completion tokens",
            "color": CHART_COLORS["tokens"],
            "higher_is_better": True,
            "number_format": "{:,.0f}",
        },
    ]

    for category, folder_name in TASK_FOLDERS.items():
        models = task_metrics[category]
        for spec in chart_specs:
            values_by_model = {
                model_name: metrics[spec["metric"]] for model_name, metrics in models.items()
            }
            render_bar_chart(
                values_by_model=values_by_model,
                output_path=SLIDE_DIR / folder_name / spec["filename"],
                title=f"{spec['title']} - {category}",
                x_label=spec["x_label"],
                color=spec["color"],
                higher_is_better=spec["higher_is_better"],
                number_format=spec["number_format"],
            )

    overall_scores: dict[str, list[float]] = defaultdict(list)
    overall_latencies: dict[str, list[float]] = defaultdict(list)
    overall_tokens: dict[str, list[float]] = defaultdict(list)
    raw_metrics = load_raw_metrics()
    for row in judge_rows:
        model_name = row["model_name"]
        raw = raw_metrics[(model_name, row["sample_id"])]
        overall_scores[model_name].append(float(row["weighted_score"]))
        overall_latencies[model_name].append(raw["latency_s"])
        overall_tokens[model_name].append(raw["completion_tokens"])
    average_scores = {
        model_name: mean(scores) for model_name, scores in overall_scores.items()
    }
    average_latencies = {
        model_name: mean(latencies) for model_name, latencies in overall_latencies.items()
    }
    average_tokens = {
        model_name: mean(tokens) for model_name, tokens in overall_tokens.items()
    }
    render_bar_chart(
        values_by_model=average_scores,
        output_path=SLIDE_DIR / "00_overall_model_comparison.png",
        title="Overall Model Quality Comparison - 10 Questions",
        x_label="Average weighted score / 10",
        color=CHART_COLORS["score"],
        higher_is_better=True,
        number_format="{:.2f}",
    )
    render_bar_chart(
        values_by_model=average_latencies,
        output_path=SLIDE_DIR / "01_overall_response_time_comparison.png",
        title="Overall Response Time Comparison - 10 Questions",
        x_label="Average response time (seconds)",
        color=CHART_COLORS["latency"],
        higher_is_better=False,
        number_format="{:.1f}s",
        footer_note=(
            "Phi-3 runs locally; most other models use API providers. "
            "Latency is observational, not a hardware-normalized comparison."
        ),
    )
    render_bar_chart(
        values_by_model=average_tokens,
        output_path=SLIDE_DIR / "02_overall_generated_tokens_comparison.png",
        title="Overall Generated Tokens Comparison - 10 Questions",
        x_label="Average completion tokens",
        color=CHART_COLORS["tokens"],
        higher_is_better=True,
        number_format="{:,.0f}",
    )
    render_quality_latency_tradeoff(
        values_by_model={
            model_name: {
                "score": average_scores[model_name],
                "latency_s": average_latencies[model_name],
            }
            for model_name in average_scores
        },
        output_path=SLIDE_DIR / "03_quality_latency_tradeoff.png",
    )
    overall_tradeoff_values = {
        model_name: {
            "score": average_scores[model_name],
            "latency_s": average_latencies[model_name],
            "completion_tokens": average_tokens[model_name],
        }
        for model_name in average_scores
    }
    render_tradeoff_scatter(
        values_by_model=overall_tradeoff_values,
        output_path=SLIDE_DIR / "04_response_time_vs_generated_tokens.png",
        title="Response Time vs. Generated Tokens - 10 Questions",
        x_metric="completion_tokens",
        y_metric="latency_s",
        x_label="Average completion tokens",
        y_label="Average response time (seconds)",
        footer_note=(
            "Phi-3 Medium stands out: relatively short output but high local inference latency. "
            "Phi-3 runs locally; most other models use API providers."
        ),
    )
    render_tradeoff_scatter(
        values_by_model=overall_tradeoff_values,
        output_path=SLIDE_DIR / "05_quality_vs_generated_tokens.png",
        title="Quality vs. Generated Tokens - 10 Questions",
        x_metric="completion_tokens",
        y_metric="score",
        x_label="Average completion tokens",
        y_label="Average weighted score / 10",
        footer_note=(
            "Upper-left is better: higher quality with a more concise response. "
            "Thinking models tend to generate longer outputs."
        ),
        y_lower=4,
        y_upper_factor=1.12,
        label_offsets={
            "Qwen3-14B": (7, -14),
            "Qwen3-32B": (7, 8),
            "Llama 4 Scout": (7, 8),
            "Phi-3 Medium 4K": (7, -14),
        },
    )

    print(f"Generated slide charts: {SLIDE_DIR}")
    print(f"- Overall charts: {SLIDE_DIR}")
    for category, folder_name in TASK_FOLDERS.items():
        print(f"- {category}: {SLIDE_DIR / folder_name}")


if __name__ == "__main__":
    main()
