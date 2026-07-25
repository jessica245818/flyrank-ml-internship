"""Repeatable, public-safe analysis for the Refresh Opportunity capstone.

The script reads only two warehouse month partitions. March is the development
month; June is opened once as the sealed temporal test. No raw data is written.
Only aggregate metrics, pseudonymized recommendations, and SVG charts are saved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "work" / "outputs" / "capstone_results.json"
CHART_DIR = ROOT / "docs" / "assets"
REL = "hf://datasets/FlyRank/internship-warehouse"
FEATURES = [
    "impressions_pre",
    "clicks_pre",
    "ctr_pre",
    "avg_position_pre",
    "active_days_pre",
]


def precision_at_k(labels: pd.Series, scores: np.ndarray | pd.Series, k: int = 50) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.asarray(labels)[order].mean())


def month_frame(con: duckdb.DuckDBPyConnection, month: str, last_day: int) -> pd.DataFrame:
    """Aggregate one month into pre-decision features and a later outcome."""
    source = (
        f"read_parquet('{REL}/fact_content_daily_performance/"
        f"month={month}/*.parquet')"
    )
    post_days = last_day - 15
    query = f"""
    WITH per_content AS (
        SELECT client_hash_id,
               content_hash_id,
               SUM(CASE WHEN DAY(report_date) <= 15
                        THEN gsc_impressions ELSE 0 END) AS impressions_pre,
               SUM(CASE WHEN DAY(report_date) <= 15
                        THEN gsc_clicks ELSE 0 END) AS clicks_pre,
               AVG(CASE WHEN DAY(report_date) <= 15 AND gsc_impressions > 0
                        THEN gsc_avg_position END) AS avg_position_pre,
               COUNT(DISTINCT CASE WHEN DAY(report_date) <= 15
                                    AND gsc_impressions > 0
                                   THEN report_date END) AS active_days_pre,
               SUM(CASE WHEN DAY(report_date) > 15
                        THEN gsc_impressions ELSE 0 END) AS impressions_post
        FROM {source}
        WHERE gsc_data_available IS TRUE
        GROUP BY 1, 2
    )
    SELECT client_hash_id,
           content_hash_id,
           impressions_pre,
           clicks_pre,
           CASE WHEN impressions_pre > 0
                THEN 100.0 * clicks_pre / impressions_pre ELSE 0 END AS ctr_pre,
           avg_position_pre,
           active_days_pre,
           impressions_post,
           CAST((impressions_post / {post_days}.0)
                < 0.8 * (impressions_pre / 15.0) AS INTEGER) AS declined_post
    FROM per_content
    WHERE impressions_pre >= 100
    """
    frame = con.sql(query).df().dropna(subset=FEATURES + ["declined_post"])
    if not frame["content_hash_id"].is_unique:
        raise ValueError(f"{month} frame violates one-row-per-content grain")
    return frame


def baseline_score(frame: pd.DataFrame) -> pd.Series:
    """Transparent opportunity score using only the five pre-decision fields."""
    visibility = np.log1p(frame["impressions_pre"]).rank(pct=True)
    position_opportunity = (
        1 - (frame["avg_position_pre"].clip(1, 50) - 1) / 49
    ) * visibility
    active = frame["active_days_pre"].rank(pct=True)
    ctr_gap = 1 - frame["ctr_pre"].rank(pct=True)
    return (
        0.45 * visibility
        + 0.25 * position_opportunity
        + 0.20 * active
        + 0.10 * ctr_gap
    ).clip(0, 1)


def reason_and_action(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    if row["impressions_pre"] >= 1_000:
        reasons.append("high_recent_visibility")
    if 0 < row["avg_position_pre"] <= 20 and row["ctr_pre"] < 0.5:
        reasons.append("low_ctr_for_visible_page")
    if row["avg_position_pre"] > 50:
        reasons.append("weak_position_context")
    if row.get("active_days_pre", 0) >= 12:
        reasons.append("persistent_observation")
    if row["model_score"] >= 0.70:
        reasons.append("high_model_decline_risk")
    if not reasons:
        reasons.append("model_review_candidate")

    if "low_ctr_for_visible_page" in reasons:
        action = "review_title_meta_and_intent_match"
    elif "weak_position_context" in reasons:
        action = "monitor_position_and_assess_demand"
    elif row["impressions_pre"] >= 1_000:
        action = "review_for_refresh_or_protection"
    else:
        action = "monitor_and_review_context"
    return "|".join(reasons), action


def save_charts(results: dict, recommendations: pd.DataFrame) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    labels = ["Transparent baseline", "Random forest"]
    values = [
        results["metrics"]["baseline"]["precision_at_50"],
        results["metrics"]["model"]["precision_at_50"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=["#94a3b8", "#16a34a"], width=0.58)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Precision@50 on sealed June test")
    ax.set_title("Top-50 review precision: model vs baseline")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "capstone_precision.svg", format="svg")
    plt.close(fig)

    importance = pd.Series(
        results["feature_importance"], name="importance"
    ).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    importance.plot.barh(ax=ax, color="#2563eb")
    ax.set_xlabel("Random-forest impurity importance")
    ax.set_title("Which pre-decision signals shaped the ranking?")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "capstone_importance.svg", format="svg")
    plt.close(fig)

    action_counts = recommendations["suggested_action"].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    action_counts.sort_values().plot.barh(ax=ax, color="#f59e0b")
    ax.set_xlabel("Items in top 50")
    ax.set_title("Recommended review actions in the final queue")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "capstone_actions.svg", format="svg")
    plt.close(fig)


def run() -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set a Hugging Face READ token in HF_TOKEN")

    con = duckdb.connect()
    safe_token = token.replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{safe_token}')"
    )
    del safe_token

    march = month_frame(con, "2026-03", 31)
    june = month_frame(con, "2026-06", 30)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(march[FEATURES], march["declined_post"])
    model_score = model.predict_proba(june[FEATURES])[:, 1]
    rule_score = baseline_score(june)

    metrics = {
        "baseline": {
            "roc_auc": roc_auc_score(june["declined_post"], rule_score),
            "average_precision": average_precision_score(
                june["declined_post"], rule_score
            ),
            "precision_at_50": precision_at_k(
                june["declined_post"], rule_score, 50
            ),
        },
        "model": {
            "roc_auc": roc_auc_score(june["declined_post"], model_score),
            "average_precision": average_precision_score(
                june["declined_post"], model_score
            ),
            "precision_at_50": precision_at_k(
                june["declined_post"], model_score, 50
            ),
        },
    }

    ranked = june[
        ["content_hash_id", "client_hash_id"] + FEATURES + ["declined_post"]
    ].copy()
    ranked["model_score"] = model_score
    ranked["baseline_score"] = np.asarray(rule_score)
    ranked = ranked.sort_values(
        ["model_score", "impressions_pre"], ascending=[False, False]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    explanations = ranked.apply(reason_and_action, axis=1)
    ranked["reason_codes"] = [item[0] for item in explanations]
    ranked["suggested_action"] = [item[1] for item in explanations]
    top50 = ranked.head(50)

    results = {
        "release": "flyrank_pseudonymized_warehouse_release_v20260703",
        "lane": "Refresh / Content Opportunity Scoring",
        "development_window": "2026-03-01 to 2026-03-31",
        "sealed_test_window": "2026-06-01 to 2026-06-30",
        "decision_day": 15,
        "features": FEATURES,
        "development_rows": int(len(march)),
        "test_rows": int(len(june)),
        "development_clients": int(march["client_hash_id"].nunique()),
        "test_clients": int(june["client_hash_id"].nunique()),
        "test_positive_rate": float(june["declined_post"].mean()),
        "metrics": metrics,
        "feature_importance": dict(
            zip(FEATURES, [float(x) for x in model.feature_importances_])
        ),
        "top_recommendations": [
            {
                "rank": int(row["rank"]),
                "content_hash_id": row["content_hash_id"],
                "model_score": float(row["model_score"]),
                "impressions_pre": int(row["impressions_pre"]),
                "ctr_pre": float(row["ctr_pre"]),
                "avg_position_pre": float(row["avg_position_pre"]),
                "active_days_pre": int(row["active_days_pre"]),
                "reason_codes": row["reason_codes"],
                "suggested_action": row["suggested_action"],
            }
            for _, row in top50.iterrows()
        ],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    save_charts(results, top50)
    return results


if __name__ == "__main__":
    output = run()
    print(json.dumps({
        "development_rows": output["development_rows"],
        "test_rows": output["test_rows"],
        "test_positive_rate": output["test_positive_rate"],
        "metrics": output["metrics"],
    }, indent=2))
