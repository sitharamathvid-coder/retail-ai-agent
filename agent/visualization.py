import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

# --- UTILITIES ---

def humanize_column_name(name: str) -> str:
    """Converts 'total_revenue' to 'Revenue', 'freight_cost' to 'Freight Cost'."""
    if not name:
        return name
    cleaned = name.replace("total_", "").replace("_", " ")
    return " ".join(word.capitalize() for word in cleaned.split())

def format_number(value: float, is_currency: bool = False, is_percentage: bool = False) -> str:
    """Formats a number for tooltips/annotations."""
    try:
        val = float(value)
        if is_percentage:
            return f"{val:.1f}%"
        
        abs_val = abs(val)
        prefix = "R$ " if is_currency else ""
        
        if abs_val >= 1_000_000_000:
            return f"{prefix}{val / 1_000_000_000:.2f}B"
        if abs_val >= 1_000_000:
            return f"{prefix}{val / 1_000_000:.2f}M"
        if abs_val >= 1_000:
            return f"{prefix}{val / 1_000:.1f}K"
        
        return f"{prefix}{val:,.2f}" if is_currency else f"{val:,.0f}"
    except (ValueError, TypeError):
        return str(value)

def determine_metric_type(metric_name: str) -> Tuple[bool, bool]:
    """Returns (is_currency, is_percentage) based on column name."""
    name_lower = metric_name.lower()
    is_curr = any(kw in name_lower for kw in ["revenue", "profit", "cost", "price", "value", "payment", "freight"])
    is_pct = any(kw in name_lower for kw in ["pct", "percent", "ratio", "rate", "share"])
    return is_curr, is_pct

def apply_enterprise_theme(fig: go.Figure) -> go.Figure:
    """Applies premium Tableau Pulse / Power BI Copilot dark theme."""
    fig.update_layout(
        height=480,
        margin=dict(l=24, r=24, t=80, b=40),
        title=dict(
            font=dict(size=22, family="Inter, sans-serif", color="#f8fafc"),
            x=0.02,
            xanchor="left",
            y=0.95
        ),
        font=dict(size=14, family="Inter, sans-serif", color="#cbd5e1"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            font_size=14,
            font_family="Inter, sans-serif",
            bordercolor="rgba(96, 165, 250, 0.4)",
            namelength=-1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color="#94a3b8")
        ),
        hovermode="x unified"
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        tickfont=dict(size=13, color="#94a3b8"),
        title_font=dict(size=14, color="#e2e8f0"),
        automargin=True
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        tickfont=dict(size=13, color="#94a3b8"),
        title_font=dict(size=14, color="#e2e8f0"),
        automargin=True
    )
    return fig

def add_smart_annotations(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str, is_horizontal: bool = False):
    """Automatically annotates the highest performer."""
    if df.empty or len(df) < 2:
        return

    try:
        max_idx = df[y_col].idxmax() if not is_horizontal else df[x_col].idxmax()
        max_val = df.loc[max_idx, y_col] if not is_horizontal else df.loc[max_idx, x_col]
        max_cat = df.loc[max_idx, x_col] if not is_horizontal else df.loc[max_idx, y_col]
        
        metric_type = determine_metric_type(y_col if not is_horizontal else x_col)
        formatted_val = format_number(max_val, is_currency=metric_type[0], is_percentage=metric_type[1])
        
        fig.add_annotation(
            x=max_cat if not is_horizontal else max_val,
            y=max_val if not is_horizontal else max_cat,
            text=f"👑 Top Performer<br>{formatted_val}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#38bdf8",
            font=dict(size=12, color="#38bdf8"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#38bdf8",
            borderwidth=1,
            borderpad=4,
            ax=40 if is_horizontal else 0,
            ay=0 if is_horizontal else -40
        )
    except Exception as e:
        logger.warning(f"Failed to add smart annotation: {e}")

# --- CHART BUILDERS ---

def build_horizontal_ranking(df: pd.DataFrame, cat_col: str, metric_col: str, top_n: int = 10) -> go.Figure:
    """Builds a horizontal bar chart for rankings."""
    chart_data = df.sort_values(metric_col, ascending=False).head(top_n).sort_values(metric_col, ascending=True)
    
    is_curr, is_pct = determine_metric_type(metric_col)
    
    # Custom hover text
    hover_texts = [
        f"<b>{row[cat_col]}</b><br>{humanize_column_name(metric_col)}: {format_number(row[metric_col], is_curr, is_pct)}"
        for _, row in chart_data.iterrows()
    ]
    
    fig = go.Figure(go.Bar(
        x=chart_data[metric_col],
        y=chart_data[cat_col],
        orientation="h",
        marker=dict(
            color="#2563eb",
            line=dict(color="rgba(255,255,255,0.1)", width=1),
            # Gradient approximation by opacity
            opacity=0.9
        ),
        text=[format_number(val, is_curr, is_pct) for val in chart_data[metric_col]],
        textposition="outside",
        hoverinfo="text",
        hovertext=hover_texts
    ))
    
    title = f"Top {len(chart_data)} {humanize_column_name(cat_col)} by {humanize_column_name(metric_col)}"
    fig.update_layout(title_text=title, yaxis=dict(type='category'))
    
    fig = apply_enterprise_theme(fig)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=False)
    
    add_smart_annotations(fig, chart_data, metric_col, cat_col, is_horizontal=True)
    return fig

def build_time_series(df: pd.DataFrame, date_col: str, metric_col: str) -> go.Figure:
    """Builds a smooth trend line chart."""
    chart_data = df.sort_values(date_col)
    is_curr, is_pct = determine_metric_type(metric_col)
    
    hover_texts = [
        f"<b>{row[date_col]}</b><br>{humanize_column_name(metric_col)}: {format_number(row[metric_col], is_curr, is_pct)}"
        for _, row in chart_data.iterrows()
    ]
    
    fig = go.Figure(go.Scatter(
        x=chart_data[date_col],
        y=chart_data[metric_col],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=4, shape="spline", smoothing=0.3),
        marker=dict(size=10, color="#0f172a", line=dict(color="#38bdf8", width=2)),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.1)",
        hoverinfo="text",
        hovertext=hover_texts
    ))
    
    title = f"{humanize_column_name(metric_col)} Trend over {humanize_column_name(date_col)}"
    fig.update_layout(title_text=title)
    
    fig = apply_enterprise_theme(fig)
    add_smart_annotations(fig, chart_data, date_col, metric_col)
    return fig

def build_donut_chart(df: pd.DataFrame, cat_col: str, metric_col: Optional[str] = None) -> Optional[go.Figure]:
    """Builds a share donut chart if categories are small."""
    if metric_col:
        chart_data = df.sort_values(metric_col, ascending=False)
        val_col = metric_col
    else:
        chart_data = df[cat_col].value_counts().reset_index()
        chart_data.columns = [cat_col, "count"]
        val_col = "count"
        
    if len(chart_data) > 8:
        logger.info("Donut chart rejected due to excessive category count (>8).")
        return None
        
    is_curr, is_pct = determine_metric_type(val_col)
    
    fig = go.Figure(go.Pie(
        labels=chart_data[cat_col],
        values=chart_data[val_col],
        hole=0.6,
        marker=dict(
            colors=px.colors.qualitative.Prism,
            line=dict(color="#0f172a", width=2)
        ),
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>" + humanize_column_name(val_col) + ": %{value}<br>Share: %{percent}<extra></extra>"
    ))
    
    title = f"{humanize_column_name(val_col)} Distribution by {humanize_column_name(cat_col)}"
    fig.update_layout(title_text=title)
    fig = apply_enterprise_theme(fig)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig

def build_dual_axis_combo(df: pd.DataFrame, cat_col: str, metric1: str, metric2: str, top_n: int = 10) -> go.Figure:
    """Builds a combo bar+line chart with intelligent relationship ratios."""
    chart_data = df.sort_values(metric1, ascending=False).head(top_n)
    
    is_curr1, is_pct1 = determine_metric_type(metric1)
    is_curr2, is_pct2 = determine_metric_type(metric2)
    
    # Compute ratio if meaningful (avoid div 0)
    has_ratio = False
    ratios = []
    for _, row in chart_data.iterrows():
        val1 = row[metric1]
        val2 = row[metric2]
        if val1 and val1 > 0:
            ratio = (val2 / val1) * 100
            ratios.append(ratio)
            has_ratio = True
        else:
            ratios.append(0)
            
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar trace (Primary)
    hover_bar = [
        f"<b>{row[cat_col]}</b><br>{humanize_column_name(metric1)}: {format_number(row[metric1], is_curr1, is_pct1)}"
        + (f"<br>{humanize_column_name(metric2)} to {humanize_column_name(metric1)} Ratio: {ratios[i]:.1f}%" if has_ratio else "")
        for i, row in chart_data.reset_index().iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=chart_data[cat_col],
            y=chart_data[metric1],
            name=humanize_column_name(metric1),
            marker=dict(color="#2563eb", opacity=0.85, line=dict(width=0)),
            hoverinfo="text",
            hovertext=hover_bar
        ),
        secondary_y=False,
    )
    
    # Spline trace (Secondary)
    hover_line = [
        f"<b>{row[cat_col]}</b><br>{humanize_column_name(metric2)}: {format_number(row[metric2], is_curr2, is_pct2)}"
        for _, row in chart_data.iterrows()
    ]
    fig.add_trace(
        go.Scatter(
            x=chart_data[cat_col],
            y=chart_data[metric2],
            name=humanize_column_name(metric2),
            mode="lines+markers",
            line=dict(color="#38bdf8", width=4, shape="spline", smoothing=0.3),
            marker=dict(size=12, color="#0f172a", line=dict(color="#38bdf8", width=2)),
            hoverinfo="text",
            hovertext=hover_line
        ),
        secondary_y=True,
    )
    
    title = f"{humanize_column_name(metric1)} vs {humanize_column_name(metric2)} by Top {humanize_column_name(cat_col)}"
    fig.update_layout(title_text=title)
    
    fig = apply_enterprise_theme(fig)
    fig.update_yaxes(title_text=humanize_column_name(metric1), secondary_y=False)
    fig.update_yaxes(title_text=humanize_column_name(metric2), secondary_y=True, showgrid=False)
    return fig

# --- INTELLIGENT SELECTOR ---

def get_column_groups(dataframe: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = dataframe.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_columns = [
        col for col in dataframe.columns
        if col not in numeric_columns and col not in datetime_columns
    ]
    return categorical_columns, numeric_columns, datetime_columns

def select_and_build_chart(
    df: pd.DataFrame, 
    override_type: Optional[str] = "Auto", 
    top_n: int = 10,
    selected_metrics: Optional[List[str]] = None
) -> Optional[go.Figure]:
    """
    Intelligently selects and builds the best enterprise chart for the dataframe.
    """
    if df is None or df.empty or len(df.columns) < 2:
        logger.info("Visualization skipped: insufficient data or columns.")
        return None
        
    cat_cols, num_cols, date_cols = get_column_groups(df)
    
    if selected_metrics:
        num_cols = [m for m in selected_metrics if m in num_cols]
    
    # 1. Trend Analysis
    if date_cols and num_cols and (override_type == "Auto" or override_type == "Line"):
        logger.info("Selected time-series line chart for trend visualization.")
        return build_time_series(df, date_cols[0], num_cols[0])
        
    # 2. Categorical Analysis
    if cat_cols and num_cols:
        cat_col = cat_cols[0]
        
        # Override to Donut
        if override_type == "Pie":
            fig = build_donut_chart(df, cat_col, num_cols[0])
            if fig: return fig
            
        # Multi-metric Combo
        if len(num_cols) >= 2 and override_type == "Auto":
            logger.info("Selected dual-axis combo chart for multi-metric comparison.")
            return build_dual_axis_combo(df, cat_col, num_cols[0], num_cols[1], top_n)
            
        # Standard Ranking
        if override_type in ["Auto", "Bar"]:
            unique_cats = df[cat_col].nunique()
            if unique_cats <= 6 and override_type == "Auto":
                # Try donut first for small sets
                fig = build_donut_chart(df, cat_col, num_cols[0])
                if fig:
                    logger.info("Selected donut chart for small category distribution.")
                    return fig
            
            logger.info("Selected horizontal bar chart for ranking visualization.")
            return build_horizontal_ranking(df, cat_col, num_cols[0], top_n)
            
    # 3. Pure categorical distribution (no metrics)
    if not num_cols and cat_cols:
        cat_col = cat_cols[0]
        if df[cat_col].nunique() <= 8 and override_type in ["Auto", "Pie"]:
            logger.info("Selected donut chart for pure categorical distribution.")
            return build_donut_chart(df, cat_col, None)
            
    logger.info("No suitable chart pattern found for the result set.")
    return None
