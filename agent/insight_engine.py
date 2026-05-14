import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def format_currency(value: float) -> str:
    """
    Formats large monetary values into concise executive format.
    Example: 229472.63 -> R$ 229.5K
    """
    try:
        absolute_value = abs(float(value))
        if absolute_value >= 1_000_000_000:
            return f"R$ {value / 1_000_000_000:.2f}B"
        if absolute_value >= 1_000_000:
            return f"R$ {value / 1_000_000:.2f}M"
        if absolute_value >= 1_000:
            return f"R$ {value / 1_000:.1f}K"
        return f"R$ {value:,.2f}"
    except (ValueError, TypeError):
        return str(value)

def format_large_number(value: float) -> str:
    """
    Formats large numeric counts into concise executive format.
    Example: 1120000 -> 1.12M
    """
    try:
        absolute_value = abs(float(value))
        if absolute_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if absolute_value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if absolute_value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:,.0f}"
    except (ValueError, TypeError):
        return str(value)

def format_percentage(value: float) -> str:
    """
    Formats a float representing a proportion or percentage into string.
    Example: 0.6123 -> 61.2% (if it represents 61.2%), or if given as 61.23, -> 61.2%
    Assuming input is already a percentage (0-100).
    """
    try:
        return f"{float(value):.1f}%"
    except (ValueError, TypeError):
        return str(value)

def clean_column_name(name: str) -> str:
    """Converts raw database column names into readable titles."""
    if not name:
        return name
    cleaned = name.replace("_", " ")
    return " ".join(word.capitalize() for word in cleaned.split())

def extract_insight_metadata(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Automatically detect primary entity column and primary metric column.
    Priority: revenue, profit, sales, order_count, customers, quantity
    """
    result = {"entity_col": None, "metric_col": None}
    if df is None or df.empty:
        return result

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if cat_cols:
        result["entity_col"] = cat_cols[0]

    if numeric_cols:
        priority_list = ["revenue", "profit", "sales", "order_count", "customers", "quantity", "price", "value", "payment", "count", "total"]
        metric_col = None
        for p in priority_list:
            for col in numeric_cols:
                if p in col.lower():
                    metric_col = col
                    break
            if metric_col:
                break
        if not metric_col:
            metric_col = numeric_cols[0]
        result["metric_col"] = metric_col

    return result

def detect_concentration_risk(df: pd.DataFrame, metric_col: str) -> Dict[str, Any]:
    """
    Calculate concentration dominance using Pareto-style analysis.
    """
    if df is None or df.empty or metric_col not in df.columns:
        return {"top_n": 0, "contribution_pct": 0.0, "risk_level": "unknown"}
    
    total_val = float(df[metric_col].sum())
    if total_val <= 0:
        return {"top_n": 0, "contribution_pct": 0.0, "risk_level": "none"}

    df_sorted = df.sort_values(by=metric_col, ascending=False)
    top_n = max(3, int(len(df) * 0.2))
    top_n = min(top_n, len(df))
    
    top_sum = float(df_sorted[metric_col].head(top_n).sum())
    contribution_pct = (top_sum / total_val) * 100

    if contribution_pct >= 70:
        risk_level = "high"
    elif contribution_pct >= 40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "top_n": top_n,
        "contribution_pct": contribution_pct,
        "risk_level": risk_level
    }

def compare_top_entities(df: pd.DataFrame, entity_col: str, metric_col: str) -> Dict[str, Any]:
    """
    Compute top vs second %, top vs median %, and ranking gap classification.
    """
    if df is None or len(df) < 2 or not entity_col or not metric_col or entity_col not in df.columns or metric_col not in df.columns:
        return {"top_vs_second_pct": 0.0, "top_vs_median_pct": 0.0, "ranking_gap": "unknown"}
        
    df_sorted = df.sort_values(by=metric_col, ascending=False)
    
    top_val = float(df_sorted[metric_col].iloc[0])
    second_val = float(df_sorted[metric_col].iloc[1])
    median_val = float(df_sorted[metric_col].median())
    
    top_vs_second_pct = 0.0
    if second_val > 0:
        top_vs_second_pct = ((top_val - second_val) / second_val) * 100
        
    top_vs_median_pct = 0.0
    if median_val > 0:
        top_vs_median_pct = ((top_val - median_val) / median_val) * 100

    if top_vs_second_pct >= 50:
        ranking_gap = "significant"
    elif top_vs_second_pct >= 20:
        ranking_gap = "moderate"
    else:
        ranking_gap = "slight"

    return {
        "top_vs_second_pct": top_vs_second_pct,
        "top_vs_median_pct": top_vs_median_pct,
        "ranking_gap": ranking_gap
    }

def summarize_distribution(df: pd.DataFrame, metric_col: str) -> Dict[str, Any]:
    """
    Compute mean, median, std deviation, skewness classification, and variance level.
    """
    if df is None or df.empty or metric_col not in df.columns:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "skewness": "unknown", "variance_level": "unknown"}
        
    mean_val = float(df[metric_col].mean())
    median_val = float(df[metric_col].median())
    std_val = float(df[metric_col].std()) if len(df) > 1 else 0.0
    
    # Skewness
    skew_val = float(df[metric_col].skew()) if len(df) > 2 else 0.0
    if abs(skew_val) > 1.0:
        skew_class = "highly_skewed"
    elif abs(skew_val) > 0.5:
        skew_class = "moderately_skewed"
    else:
        skew_class = "balanced"
        
    # Variance (Coefficient of Variation)
    cv = (std_val / mean_val) if mean_val > 0 else 0.0
    if cv > 1.0:
        var_level = "high"
    elif cv > 0.5:
        var_level = "moderate"
    else:
        var_level = "low"

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "skewness": skew_class,
        "variance_level": var_level,
        "total": float(df[metric_col].sum())
    }

def generate_statistical_context(df: pd.DataFrame) -> str:
    """
    Generate deterministic context text for the LLM.
    """
    try:
        if df is None or df.empty:
            return "No data available to generate statistical context."

        metadata = extract_insight_metadata(df)
        entity_col = metadata.get("entity_col")
        metric_col = metadata.get("metric_col")
        
        if not metric_col:
            return "No valid numeric metric column found for statistical analysis."

        metric_name = clean_column_name(metric_col)
        entity_name = clean_column_name(entity_col) if entity_col else "entities"

        concentration = detect_concentration_risk(df, metric_col)
        distribution = summarize_distribution(df, metric_col)
        
        context_lines = [
            "Computed Statistical Context:",
            f"- Total {metric_name}: {format_currency(distribution['total']) if 'revenue' in metric_col.lower() or 'profit' in metric_col.lower() else format_large_number(distribution['total'])}",
            f"- Median {metric_name}: {format_currency(distribution['median']) if 'revenue' in metric_col.lower() or 'profit' in metric_col.lower() else format_large_number(distribution['median'])}"
        ]
        
        if concentration['top_n'] > 0:
            context_lines.append(f"- Top {concentration['top_n']} {entity_name} contribute {format_percentage(concentration['contribution_pct'])} of total {metric_name}")
        
        context_lines.append(f"- {metric_name} distribution is {distribution['skewness'].replace('_', ' ')}")
        
        if entity_col:
            comparisons = compare_top_entities(df, entity_col, metric_col)
            if comparisons['top_vs_median_pct'] > 0:
                context_lines.append(f"- Top entity outperformed median performers by {format_percentage(comparisons['top_vs_median_pct'])}")
            if comparisons['top_vs_second_pct'] > 0:
                context_lines.append(f"- Ranking gap between #1 and #2 is {comparisons['ranking_gap']} ({format_percentage(comparisons['top_vs_second_pct'])} difference)")
                
        if concentration['risk_level'] != "none":
            context_lines.append(f"- {concentration['risk_level'].capitalize()} concentration risk detected")

        logger.info("Statistical intelligence analysis completed.")
        return "\n".join(context_lines)
        
    except Exception as exc:
        logger.exception("Failed to generate statistical context.")
        return f"Statistical context generation failed: {exc}"

# Lightweight Unit Test Examples:
# if __name__ == "__main__":
#     df_test = pd.DataFrame({
#         "seller": ["A", "B", "C", "D", "E"],
#         "total_revenue": [100000, 50000, 10000, 5000, 1000]
#     })
#     print(generate_statistical_context(df_test))
