from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_judge import METRIC_WEIGHTS, dump_json, summarize, weighted_score, write_csv


JUDGE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_judge"
RESULTS_PATH = JUDGE_OUTPUT_DIR / "results" / "llm_judge_results.json"
SUMMARY_PATH = JUDGE_OUTPUT_DIR / "results" / "llm_judge_summary.json"
TABLES_DIR = JUDGE_OUTPUT_DIR / "tables"
VISUALS_DIR = JUDGE_OUTPUT_DIR / "visuals"

NON_CODE_RATIONALE = "Not applicable: task does not request code or SQL."
MAVERICK_RATIONALE = "Provides practical guidance for table lifecycle and production usage."
MAVERICK_SAMPLE_ID = "BD_THEORY_EASY_02"
MAVERICK_MODEL_NAME = "Llama 4 Maverick"


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def fix_results(results: list[dict[str, Any]]) -> tuple[int, int]:
    code_sql_fixes = 0
    maverick_text_fixes = 0

    for row in results:
        judge = row.get("judge") or {}
        scores = judge.get("scores") or {}
        code_sql = scores.get("code_sql_quality")

        if (
            row.get("category") != "Code Generation"
            and isinstance(code_sql, dict)
            and code_sql.get("applicable", True)
        ):
            code_sql.update(
                {
                    "applicable": False,
                    "score": None,
                    "rationale": NON_CODE_RATIONALE,
                }
            )
            row["weighted_score"] = weighted_score(scores)
            code_sql_fixes += 1

        if (
            row.get("sample_id") == MAVERICK_SAMPLE_ID
            and row.get("model_name") == MAVERICK_MODEL_NAME
        ):
            production = scores.get("production_readiness")
            if (
                isinstance(production, dict)
                and (
                    not production.get("applicable")
                    or production.get("score") != 9
                    or production.get("rationale") != MAVERICK_RATIONALE
                )
            ):
                production.update(
                    {
                        "applicable": True,
                        "score": 9,
                        "rationale": MAVERICK_RATIONALE,
                    }
                )
                row["weighted_score"] = weighted_score(scores)
                maverick_text_fixes += 1

    return code_sql_fixes, maverick_text_fixes


def build_frames(results: list[dict[str, Any]]):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to refresh judge tables. Install pandas, matplotlib, and seaborn."
        ) from exc

    rows = []
    metric_rows = []
    metric_names = list(METRIC_WEIGHTS)

    for row in results:
        judge = row.get("judge") or {}
        scores = judge.get("scores") or {}
        output = row.get("candidate_output") or ""
        base = {
            "sample_id": row.get("sample_id"),
            "model_name": row.get("model_name") or "unknown",
            "model_id": row.get("model_id"),
            "category": row.get("category") or "unknown",
            "difficulty": row.get("difficulty") or "unknown",
            "topic": row.get("topic"),
            "weighted_score": row.get("weighted_score"),
            "judge_overall_score": judge.get("overall_score"),
            "verdict": judge.get("verdict"),
            "judge_error": row.get("judge_error"),
            "output_chars": len(output),
            "output_words": len(output.split()),
            "missing_count": len(judge.get("missing_or_weak_points") or []),
            "issue_count": len(judge.get("factual_or_logic_issues") or []),
        }
        for metric in metric_names:
            metric_value = scores.get(metric) or {}
            applicable = metric_value.get("applicable", True)
            base[metric] = metric_value.get("score") if applicable else None
            metric_rows.append(
                {
                    **{
                        key: base[key]
                        for key in ["sample_id", "model_name", "category", "difficulty"]
                    },
                    "metric": metric,
                    "score": base[metric],
                    "applicable": applicable,
                    "rationale": metric_value.get("rationale"),
                }
            )
        rows.append(base)

    dashboard = pd.DataFrame(rows)
    metric_long = pd.DataFrame(metric_rows)
    scored = dashboard[dashboard["weighted_score"].notna()].copy()
    leaderboard = (
        scored.groupby("model_name", dropna=False)
        .agg(
            count=("weighted_score", "size"),
            avg_score=("weighted_score", "mean"),
            median_score=("weighted_score", "median"),
            min_score=("weighted_score", "min"),
            max_score=("weighted_score", "max"),
            avg_missing=("missing_count", "mean"),
            avg_issues=("issue_count", "mean"),
        )
        .sort_values("avg_score", ascending=False)
        .round(3)
    )
    return dashboard, metric_long, scored, leaderboard


def write_tables(results: list[dict[str, Any]], dashboard, metric_long, scored, leaderboard) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(results, TABLES_DIR / "llm_judge_results.csv")
    scored.to_csv(TABLES_DIR / "llm_dashboard_table.csv", index=False, encoding="utf-8")
    metric_long.to_csv(TABLES_DIR / "llm_metric_long_table.csv", index=False, encoding="utf-8")
    leaderboard.to_csv(TABLES_DIR / "llm_model_leaderboard.csv", encoding="utf-8")


def model_family(model_name: str) -> str:
    if model_name.startswith("Llama"):
        return "llama4"
    if model_name.startswith("Phi"):
        return "phi3"
    if model_name.startswith("Qwen"):
        return "qwen3"
    return "other"


def render_visuals(dashboard, metric_long) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and seaborn are required to refresh judge visuals. "
            "Install pandas, matplotlib, and seaborn, or pass --skip-visuals."
        ) from exc

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 10

    frames = {"overall": (dashboard, metric_long)}
    for family in ["llama4", "phi3", "qwen3"]:
        selected_models = {
            name for name in dashboard["model_name"].unique() if model_family(name) == family
        }
        frames[family] = (
            dashboard[dashboard["model_name"].isin(selected_models)].copy(),
            metric_long[metric_long["model_name"].isin(selected_models)].copy(),
        )

    for family, (family_dashboard, family_metrics) in frames.items():
        render_family_visuals(
            plt=plt,
            pd=pd,
            sns=sns,
            visual_dir=VISUALS_DIR / family,
            dashboard=family_dashboard,
            metric_long=family_metrics,
        )


def render_family_visuals(*, plt, pd, sns, visual_dir: Path, dashboard, metric_long) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)
    scored = dashboard[dashboard["weighted_score"].notna()].copy()
    if scored.empty:
        return

    def savefig(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(visual_dir / name, bbox_inches="tight")
        plt.close(fig)

    leaderboard = (
        scored.groupby("model_name", dropna=False)
        .agg(avg_score=("weighted_score", "mean"))
        .sort_values("avg_score", ascending=False)
        .round(3)
    )
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(leaderboard))))
    plot_df = leaderboard.reset_index().sort_values("avg_score")
    sns.barplot(data=plot_df, y="model_name", x="avg_score", ax=ax, color="#4c78a8")
    ax.set_xlim(0, 10)
    ax.set_title("Average Weighted Score by Model")
    ax.set_xlabel("Score / 10")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    savefig(fig, "01_model_leaderboard.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(
        data=scored,
        x="weighted_score",
        hue="model_name",
        bins=10,
        multiple="layer",
        ax=axes[0],
    )
    axes[0].set_xlim(0, 10)
    axes[0].set_title("Score Distribution")
    axes[0].set_xlabel("Weighted score / 10")
    sns.boxplot(data=scored, x="weighted_score", y="model_name", ax=axes[1])
    axes[1].set_xlim(0, 10)
    axes[1].set_title("Score Spread by Model")
    axes[1].set_xlabel("Weighted score / 10")
    axes[1].set_ylabel("")
    savefig(fig, "02_score_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=scored,
        x="difficulty",
        y="weighted_score",
        hue="model_name",
        order=["easy", "medium", "hard"],
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_ylim(0, 10)
    axes[0].set_title("Average Score by Difficulty")
    axes[0].set_xlabel("Difficulty")
    axes[0].set_ylabel("Score / 10")
    category = (
        scored.groupby(["category", "model_name"], as_index=False)["weighted_score"]
        .mean()
        .sort_values("weighted_score")
    )
    sns.barplot(
        data=category,
        y="category",
        x="weighted_score",
        hue="model_name",
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_xlim(0, 10)
    axes[1].set_title("Average Score by Category")
    axes[1].set_xlabel("Score / 10")
    axes[1].set_ylabel("")
    savefig(fig, "03_score_by_difficulty_category.png")

    pivot_category = scored.pivot_table(
        index="category",
        columns="model_name",
        values="weighted_score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(
            max(7, 1.2 * len(pivot_category.columns)),
            max(4, 0.45 * len(pivot_category)),
        )
    )
    sns.heatmap(
        pivot_category,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=10,
        ax=ax,
    )
    ax.set_title("Average Weighted Score: Category x Model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig(fig, "04_category_model_heatmap.png")

    metric_names = list(METRIC_WEIGHTS)
    metric_scored = metric_long[metric_long["score"].notna()].copy()
    pivot_metric = metric_scored.pivot_table(
        index="metric",
        columns="model_name",
        values="score",
        aggfunc="mean",
    ).reindex(metric_names)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(pivot_metric.columns)), 6))
    sns.heatmap(
        pivot_metric,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=10,
        ax=ax,
    )
    ax.set_title("Rubric Metric Averages by Model")
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig(fig, "05_metric_model_heatmap.png")

    radar_metrics = [
        metric
        for metric in metric_names
        if metric != "code_sql_quality" or pivot_metric.loc[metric].notna().any()
    ]
    angles = [n / float(len(radar_metrics)) * 2 * math.pi for n in range(len(radar_metrics))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for name in pivot_metric.columns:
        values = [pivot_metric.loc[metric, name] for metric in radar_metrics]
        values = [0 if pd.isna(value) else float(value) for value in values]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace("_", "\n") for metric in radar_metrics], fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_title("Metric Profile Radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    savefig(fig, "06_metric_radar.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sns.scatterplot(
        data=scored,
        x="output_words",
        y="weighted_score",
        hue="model_name",
        style="difficulty",
        s=80,
        ax=axes[0],
    )
    axes[0].set_ylim(0, 10)
    axes[0].set_title("Score vs Output Length")
    axes[0].set_xlabel("Output words")
    axes[0].set_ylabel("Score / 10")
    sns.barplot(data=scored, x="model_name", y="missing_count", errorbar=None, ax=axes[1])
    axes[1].set_title("Avg Missing/Weak Points")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    sns.barplot(data=scored, x="model_name", y="issue_count", errorbar=None, ax=axes[2])
    axes[2].set_title("Avg Factual/Logic Issues")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=25)
    savefig(fig, "07_length_missing_issues.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply narrow manual rubric corrections and refresh LLM judge artifacts."
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Refresh JSON and CSV files without regenerating PNG visuals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(RESULTS_PATH)
    code_sql_fixes, maverick_text_fixes = fix_results(results)

    dump_json(results, RESULTS_PATH)
    dump_json(summarize(results), SUMMARY_PATH)
    dashboard, metric_long, scored, leaderboard = build_frames(results)
    write_tables(results, dashboard, metric_long, scored, leaderboard)
    if not args.skip_visuals:
        render_visuals(dashboard, metric_long)

    print(f"Updated non-code code_sql_quality rows: {code_sql_fixes}")
    print(f"Updated Maverick production_readiness rationale rows: {maverick_text_fixes}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Visuals refreshed: {not args.skip_visuals}")


if __name__ == "__main__":
    main()
