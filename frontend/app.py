from __future__ import annotations

import logging
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
if 'agent.agent' in sys.modules:
    importlib.reload(sys.modules['agent.agent'])

from agent.agent import build_retail_agent, create_initial_state  # noqa: E402
from agent.insight_generator import clean_column_name  # noqa: E402
from agent.visualization import select_and_build_chart, get_column_groups  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


PAGE_TITLE = "InsightFlow AI"
PAGE_SUBTITLE = "Conversational Business Intelligence Platform"
PAGE_DESCRIPTION = (
    "Ask business questions in natural language, inspect governed SQL, and turn "
    "PostgreSQL retail data into executive-ready insights."
)
QUERY_HISTORY_KEY = "query_history"
QUESTION_INPUT_KEY = "analytics_question"
DEFAULT_CHART_ROW_LIMIT = 10
LOADING_STAGES = (
    "Prompt security validation",
    "Analyzing query intent",
    "Selecting relevant tables",
    "Inspecting schemas",
    "Generating SQL",
    "Validating query",
    "Executing analytics",
    "Generating insights",
    "Rendering charts",
    "Unsafe prompt blocked",
)
NODE_LOADING_STAGES = {
    "validate_user_prompt": ("Prompt security validation",),
    "clarify_query_intent": ("Analyzing query intent",),
    "list_available_tables": ("Selecting relevant tables",),
    "choose_relevant_tables": ("Selecting relevant tables",),
    "fetch_selected_schemas": ("Inspecting schemas",),
    "generate_sql_query": ("Generating SQL",),
    "execute_sql_query": ("Validating query", "Executing analytics"),
    "generate_business_insights": ("Generating insights",),
}
SUGGESTED_QUESTIONS = (
    (
        "Revenue analytics",
        "What are the top product categories by total revenue?",
    ),
    (
        "Customer analytics",
        "Which customer states have the highest order volume?",
    ),
    (
        "Seller analytics",
        "Which sellers generate the most revenue?",
    ),
    (
        "Payment analytics",
        "What are the top payment methods by total payment value?",
    ),
    (
        "Geographic insights",
        "Which states contribute the highest revenue and order count?",
    ),
)
WORKFLOW_STAGE_ORDER = (
    "Question received",
    "Prompt security validation",
    "Intent clarified",
    "Tables selected",
    "Schemas analyzed",
    "SQL generated",
    "Query validated",
    "Query executed",
    "Insights generated",
    "Visualization rendered",
)
KPI_QUERY = """
select
    coalesce((select sum(payment_value) from payments), 0) as total_revenue,
    coalesce((select count(*) from orders), 0) as total_orders,
    coalesce((select count(*) from customers), 0) as total_customers,
    coalesce(
        (select sum(payment_value) from payments)
        / nullif((select count(*) from orders), 0),
        0
    ) as average_order_value
"""
DATETIME_COLUMN_KEYWORDS = ("date", "time", "timestamp")
NUMERIC_METRIC_KEYWORDS = (
    "revenue",
    "total",
    "totals",
    "count",
    "counts",
    "average",
    "averages",
    "avg",
    "mean",
    "sum",
    "amount",
    "value",
    "price",
    "cost",
    "payment",
    "monetary",
)


@st.cache_resource(show_spinner=False)
def get_agent(_cache_buster=1):
    return build_retail_agent()


def run_agent(question: str) -> dict[str, Any]:
    logger.info("Processing retail analytics question: %s", question)
    agent = get_agent()
    return agent.invoke(create_initial_state(question))


def stream_agent(question: str):
    logger.info("Streaming retail analytics workflow for question: %s", question)
    agent = get_agent()
    initial_state = create_initial_state(question)
    yield None, initial_state
    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in event.items():
            yield node_name, state_update


def get_mcp_server_url() -> str:
    return os.getenv("MCP_SERVER_URL", "http://localhost:8000").rstrip("/")


def execute_mcp_query(query: str, limit: int = 1) -> dict[str, Any]:
    url = f"{get_mcp_server_url()}/query"
    logger.info("Fetching dashboard metrics from MCP query endpoint.")
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json={"query": query, "limit": limit})
        response.raise_for_status()
        return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_executive_kpis() -> dict[str, float]:
    payload = execute_mcp_query(KPI_QUERY, limit=1)
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("KPI query returned no rows.")

    row = rows[0]
    return {
        "total_revenue": float(row.get("total_revenue") or 0),
        "total_orders": float(row.get("total_orders") or 0),
        "total_customers": float(row.get("total_customers") or 0),
        "average_order_value": float(row.get("average_order_value") or 0),
    }


def format_currency(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.2f}"


def format_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:,.0f}"


def initialize_session_state() -> None:
    if QUERY_HISTORY_KEY not in st.session_state:
        st.session_state[QUERY_HISTORY_KEY] = []
    if QUESTION_INPUT_KEY not in st.session_state:
        st.session_state[QUESTION_INPUT_KEY] = ""


def add_query_history(question: str, result: dict[str, Any]) -> None:
    history_item = {
        "question": question,
        "sql_query": result.get("sql_query", ""),
        "row_count": result.get("query_result", {}).get("row_count", 0),
        "has_errors": bool(result.get("errors", [])),
    }
    st.session_state[QUERY_HISTORY_KEY].insert(0, history_item)
    st.session_state[QUERY_HISTORY_KEY] = st.session_state[QUERY_HISTORY_KEY][:10]


def render_page_header() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.20), transparent 28rem),
                linear-gradient(135deg, #07111f 0%, #0f172a 52%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
            width: 18rem !important;
            min-width: 18rem !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 18rem !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #e5e7eb;
        }
        div[data-testid="stAlert"] {
            background-color: rgba(30, 41, 59, 0.8) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .brand-header {
            border: 1px solid rgba(96, 165, 250, 0.24);
            border-radius: 18px;
            padding: 28px 30px;
            margin-bottom: 24px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.82));
            box-shadow: 0 24px 80px rgba(2, 6, 23, 0.38);
        }
        .brand-kicker {
            color: #60a5fa;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .brand-title {
            color: #f8fafc;
            font-size: clamp(2.2rem, 4vw, 4.2rem);
            font-weight: 800;
            line-height: 1;
            margin-bottom: 10px;
        }
        .brand-subtitle {
            color: #bfdbfe;
            font-size: 1.24rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .brand-description {
            color: #cbd5e1;
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.65;
        }
        div[data-testid="stVerticalBlock"] > div:has(> .stMarkdown .section-shell) {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.22);
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 18px 18px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15);
            border-color: rgba(56, 189, 248, 0.3);
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
        }
        .workflow-stage {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
            background: rgba(15, 23, 42, 0.78);
        }
        .workflow-stage-title {
            font-weight: 600;
            color: #f8fafc;
        }
        .workflow-stage-detail {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-top: 4px;
        }
        .suggestion-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 12px;
            background: rgba(15, 23, 42, 0.78);
            min-height: 92px;
        }
        .suggestion-category {
            color: #38bdf8;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .progress-panel {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 14px 16px;
            background: rgba(15, 23, 42, 0.82);
            margin: 12px 0 18px;
        }
        .progress-item {
            color: #cbd5e1;
            font-size: 0.94rem;
            margin: 5px 0;
        }
        .execution-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 14px 0 18px;
        }
        .execution-meta-item {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 7px 12px;
            background: rgba(15, 23, 42, 0.82);
            color: #cbd5e1;
            font-size: 0.86rem;
            font-weight: 600;
        }
        .insight-report {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 18px 20px;
            background: rgba(15, 23, 42, 0.78);
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.22);
            margin-bottom: 8px;
        }
        .insight-report p {
            color: #e5e7eb;
            font-size: 1rem;
            line-height: 1.72;
            margin: 0 0 0.9rem 0;
        }
        .insight-report ul {
            margin: 0.2rem 0 0.9rem 1.1rem;
            padding-left: 0.7rem;
        }
        .insight-report li {
            color: #e5e7eb;
            font-size: 1rem;
            line-height: 1.65;
            margin-bottom: 0.45rem;
        }
        .insight-report p:last-child,
        .insight-report ul:last-child {
            margin-bottom: 0;
        }
        .dashboard-footer {
            margin-top: 34px;
            padding: 18px 0 4px;
            border-top: 1px solid rgba(148, 163, 184, 0.16);
            color: #94a3b8;
            text-align: center;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .sidebar-brand {
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 18px;
            background: rgba(15, 23, 42, 0.92);
        }
        .sidebar-logo {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 800;
        }
        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.84rem;
            margin-top: 4px;
        }
        .stTextInput input {
            background: #020617;
            color: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 10px;
        }
        .stButton button, .stDownloadButton button {
            border-radius: 10px;
            border: 1px solid rgba(96, 165, 250, 0.24);
            background: linear-gradient(135deg, #2563eb, #0891b2);
            color: #f8fafc;
            font-weight: 700;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            overflow: hidden;
        }
        /* ===== CUSTOM DARK SCROLLBAR ===== */

        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }

        ::-webkit-scrollbar-track {
            background: #020617;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #2563eb, #0891b2);
            border-radius: 10px;
            border: 2px solid #020617;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #3b82f6, #06b6d4);
        }

        /* Firefox */
        * {
            scrollbar-width: thin;
            scrollbar-color: #2563eb #020617;
        }
        /* Remove top whitespace and Streamlit header */

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0rem;
        }

        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }

        .block-container {
            padding-top: 1rem !important;
        }
        /* Sidebar section headings */

        .sidebar-section-title {
            color: #E2E8F0 !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-kicker">⚡ Enterprise AI Analytics</div>
            <div class="brand-title">{PAGE_TITLE}</div>
            <div class="brand-subtitle">{PAGE_SUBTITLE}</div>
            <div class="brand-description">{PAGE_DESCRIPTION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards() -> None:
    st.markdown("### 📊 Executive Overview")

    try:
        with st.spinner("Loading executive KPIs..."):
            kpis = fetch_executive_kpis()
    except httpx.HTTPStatusError as exc:
        logger.exception("MCP rejected KPI query.")
        st.warning("Executive KPIs are unavailable because the metrics query failed.")
        with st.expander("KPI error details"):
            st.write(exc.response.text)
        return
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.exception("Unable to load executive KPIs.")
        st.warning("Executive KPIs are unavailable. Check that the MCP server and PostgreSQL are running.")
        with st.expander("KPI error details"):
            st.write(str(exc))
        return

    revenue_column, orders_column, customers_column, aov_column = st.columns(4)
    revenue_column.metric("💰 Total Revenue", format_currency(kpis["total_revenue"]))
    orders_column.metric("🧾 Total Orders", format_count(kpis["total_orders"]))
    customers_column.metric("👥 Total Customers", format_count(kpis["total_customers"]))
    aov_column.metric("📈 Average Order Value", format_currency(kpis["average_order_value"]))


def set_suggested_question(question: str) -> None:
    st.session_state[QUESTION_INPUT_KEY] = question


def render_suggested_questions() -> None:
    st.markdown("#### 💡 Suggested Analytics Questions")
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for index, (category, question) in enumerate(SUGGESTED_QUESTIONS):
        with columns[index]:
            st.markdown(
                f"""
                <div class="suggestion-card">
                    <div class="suggestion-category">{escape(category)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                question,
                key=f"suggested_question_{index}",
                on_click=set_suggested_question,
                args=(question,),
                use_container_width=True,
            )


def render_question_form() -> str | None:
    with st.container(border=True):
        with st.form("retail_analytics_question_form", clear_on_submit=False):
            st.text_input(
                "Ask a retail analytics question",
                placeholder="Example: What are the top payment methods by order volume?",
                key=QUESTION_INPUT_KEY,
            )
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=False)

    render_suggested_questions()

    if not submitted:
        return None

    cleaned_question = st.session_state[QUESTION_INPUT_KEY].strip()
    if not cleaned_question:
        st.warning("Enter a retail analytics question before submitting.")
        return None

    return cleaned_question


def render_progress_tracker() -> dict[str, Any]:
    st.markdown("#### 🤖 Workflow Progress")
    container = st.container()
    placeholders = {}
    with container:
        st.markdown('<div class="progress-panel">', unsafe_allow_html=True)
        for stage in LOADING_STAGES:
            placeholders[stage] = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    return {"placeholders": placeholders, "completed": set(), "failed": None}


def update_progress_tracker(
    tracker: dict[str, Any],
    active_stage: str | None = None,
    completed_stages: tuple[str, ...] = (),
    failed_stage: str | None = None,
) -> None:
    tracker["completed"].update(completed_stages)
    if failed_stage:
        tracker["failed"] = failed_stage

    for stage, placeholder in tracker["placeholders"].items():
        if tracker["failed"] == stage:
            icon = "❌"
            status = "Failed"
        elif stage in tracker["completed"]:
            icon = "✅"
            status = "Complete"
        elif active_stage == stage:
            icon = "🔄"
            status = "In progress"
        else:
            icon = "○"
            status = "Pending"

        placeholder.markdown(
            f'<div class="progress-item">{icon} {escape(stage)} - {status}</div>',
            unsafe_allow_html=True,
        )


def run_agent_with_progress(question: str, tracker: dict[str, Any]) -> dict[str, Any]:
    final_state = create_initial_state(question)
    update_progress_tracker(tracker, active_stage="Selecting relevant tables")

    try:
        for node_name, state_update in stream_agent(question):
            if state_update:
                final_state.update(state_update)

            if node_name is None:
                continue

            completed_stages = NODE_LOADING_STAGES.get(node_name, ())
            next_stage = get_next_loading_stage(completed_stages)
            update_progress_tracker(
                tracker,
                active_stage=next_stage,
                completed_stages=completed_stages,
            )

            if final_state.get("errors"):
                failed_stage = completed_stages[-1] if completed_stages else next_stage
                update_progress_tracker(tracker, failed_stage=failed_stage)
                break
    except Exception:
        logger.exception("Streaming agent workflow failed.")
        update_progress_tracker(tracker, failed_stage="Selecting relevant tables")
        raise

    return final_state


def get_next_loading_stage(completed_stages: tuple[str, ...]) -> str | None:
    if not completed_stages:
        return None

    last_completed_index = max(LOADING_STAGES.index(stage) for stage in completed_stages)
    next_index = last_completed_index + 1
    if next_index >= len(LOADING_STAGES):
        return None

    return LOADING_STAGES[next_index]


def is_metric_column(column_name: str) -> bool:
    normalized_name = column_name.lower()
    return any(keyword in normalized_name for keyword in NUMERIC_METRIC_KEYWORDS)


def is_datetime_candidate(column_name: str, series: pd.Series) -> bool:
    normalized_name = column_name.lower()
    has_datetime_name = any(keyword in normalized_name for keyword in DATETIME_COLUMN_KEYWORDS)

    if not has_datetime_name or is_metric_column(column_name):
        return False

    if is_datetime64_any_dtype(series):
        return True

    if is_numeric_dtype(series):
        return False

    return True


def parse_datetime_column(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    non_null_source_count = series.notna().sum()
    if non_null_source_count == 0:
        return series

    parse_success_rate = converted.notna().sum() / non_null_source_count
    if parse_success_rate < 0.8:
        return series

    return converted


def build_results_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    rows = result.get("query_result", {}).get("rows", [])
    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(rows)
    for column in dataframe.columns:
        if is_datetime_candidate(column, dataframe[column]):
            dataframe[column] = parse_datetime_column(dataframe[column])

    return dataframe


def get_column_groups(dataframe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = dataframe.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns and column not in datetime_columns
    ]
    return categorical_columns, numeric_columns, datetime_columns


def render_generated_sql(result: dict[str, Any]) -> None:
    sql_query = result.get("sql_query")
    with st.expander("🧠 Generated SQL query", expanded=False):
        if sql_query:
            st.code(sql_query, language="sql")
        else:
            st.info("No SQL query was generated.")


def render_results_table(dataframe: pd.DataFrame, result: dict[str, Any]) -> None:
    st.subheader("📋 Query Results")
    if dataframe.empty:
        st.info("The query returned no rows.")
        return

    st.dataframe(dataframe, use_container_width=True, hide_index=True)
    csv_data = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="retail_analytics_results.csv",
        mime="text/csv",
    )
    st.caption(f"{result.get('query_result', {}).get('row_count', len(dataframe))} rows returned")


def make_visualization_stage(
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": "Visualization rendered",
        "status": status,
        "details": details or {},
    }


def render_automatic_charts(dataframe: pd.DataFrame) -> dict[str, Any]:
    if dataframe.empty or len(dataframe.columns) < 2:
        return make_visualization_stage(
            "skipped",
            {"reason": "No chartable result set was returned."},
        )

    st.subheader("📊 Visual Analytics")

    try:
        categorical_columns, numeric_columns, datetime_columns = get_column_groups(dataframe)
        charts_rendered = 0

        # Interactive controls
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            override_type = st.selectbox(
                "Chart Type",
                ["Auto", "Bar", "Line", "Pie"],
                key="chart_type_override"
            )
        with col2:
            top_n = st.slider("Top N", min_value=3, max_value=50, value=10, step=1, key="top_n_slider")
        with col3:
            selected_metrics = st.multiselect(
                "Metrics",
                options=numeric_columns,
                default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns,
                key="metric_selector"
            )

        fig = select_and_build_chart(dataframe, override_type, top_n, selected_metrics)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
            charts_rendered = 1
            
            # Allow download
            import io
            buf = io.StringIO()
            fig.write_html(buf, include_plotlyjs="cdn")
            st.download_button(
                label="📥 Download Chart HTML",
                data=buf.getvalue(),
                file_name="enterprise_chart.html",
                mime="text/html"
            )

        if charts_rendered == 0:
            st.info("Visualization skipped because the result set does not match a supported chart pattern.")
            return make_visualization_stage(
                "skipped",
                {"reason": "No supported chart pattern was detected."},
            )

        return make_visualization_stage(
            "completed",
            {"charts_rendered": charts_rendered},
        )
    except Exception as exc:
        logger.exception("Automatic chart generation failed.")
        st.warning("Charts could not be generated for this result set.")
        with st.expander("Chart error details"):
            st.write(str(exc))
        return make_visualization_stage(
            "failed",
            {"error": str(exc)},
        )


def get_workflow_stages(
    result: dict[str, Any],
    visualization_stage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage_by_name = {
        stage.get("name"): stage
        for stage in result.get("workflow_stages", [])
        if stage.get("name")
    }
    if visualization_stage:
        stage_by_name[visualization_stage["name"]] = visualization_stage

    stages = []
    for stage_name in WORKFLOW_STAGE_ORDER:
        if stage_name == "Unsafe prompt blocked" and stage_name not in stage_by_name:
            continue
        if stage_name == "Prompt security validation" and "Unsafe prompt blocked" in stage_by_name:
            continue
        stages.append(
            stage_by_name.get(
                stage_name,
                {
                    "name": stage_name,
                    "status": "pending",
                    "details": {},
                },
            )
        )
    return stages


def get_stage_icon(status: str) -> str:
    return {
        "completed": "✅",
        "failed": "❌",
        "skipped": "➖",
        "pending": "○",
    }.get(status, "○")


def get_stage_detail_text(stage: dict[str, Any]) -> str:
    details = stage.get("details", {})
    elapsed = stage.get("elapsed_seconds")
    detail_parts = []

    if elapsed is not None:
        detail_parts.append(f"{elapsed}s")
    if "selected_tables" in details:
        detail_parts.append(f"Tables: {', '.join(details['selected_tables'])}")
    if "schema_table_count" in details:
        detail_parts.append(f"Schemas: {details['schema_table_count']}")
    if "rows_returned" in details:
        detail_parts.append(f"Rows: {details['rows_returned']}")
    if "charts_rendered" in details:
        detail_parts.append(f"Charts: {details['charts_rendered']}")
    if "reason" in details:
        detail_parts.append(details["reason"])
    if "error" in details:
        detail_parts.append(details["error"])

    return " | ".join(detail_parts)


def render_reasoning_pipeline(
    result: dict[str, Any],
    visualization_stage: dict[str, Any] | None = None,
) -> None:
    metadata = result.get("metadata", {})
    stages = get_workflow_stages(result, visualization_stage)

    with st.expander("🤖 AI reasoning pipeline", expanded=False):
        summary_columns = st.columns(4)
        summary_columns[0].metric("Tables", len(metadata.get("selected_tables", [])))
        summary_columns[1].metric("Rows Returned", format_count(metadata.get("rows_returned", 0)))
        summary_columns[2].metric(
            "Execution Time",
            f"{metadata.get('execution_time_seconds', 0):.2f}s",
        )
        failed_count = sum(1 for stage in stages if stage.get("status") == "failed")
        summary_columns[3].metric("Failed Stages", failed_count)

        for stage in stages:
            status = stage.get("status", "pending")
            icon = get_stage_icon(status)
            detail_text = escape(get_stage_detail_text(stage))
            stage_name = escape(stage["name"])
            status_text = escape(status.title())
            st.markdown(
                f"""
                <div class="workflow-stage">
                    <div class="workflow-stage-title">{icon} {stage_name} - {status_text}</div>
                    <div class="workflow-stage-detail">{detail_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if metadata.get("selected_tables"):
            st.write("Selected tables:", metadata["selected_tables"])
        if metadata.get("generated_sql"):
            st.code(metadata["generated_sql"], language="sql")


def render_execution_metadata(result: dict[str, Any]) -> None:
    metadata = result.get("metadata", {})
    selected_tables = metadata.get("selected_tables") or result.get("selected_tables", [])
    row_count = metadata.get("rows_returned", result.get("query_result", {}).get("row_count", 0))
    execution_time = metadata.get("execution_time_seconds", 0)
    tables_used = ", ".join(selected_tables) if selected_tables else "None"

    st.markdown(
        f"""
        <div class="execution-meta">
            <div class="execution-meta-item">Execution time: {execution_time:.2f}s</div>
            <div class="execution-meta-item">Rows returned: {format_count(row_count)}</div>
            <div class="execution-meta-item">Tables used: {escape(tables_used)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="dashboard-footer">
            Powered by LangGraph • FastAPI • PostgreSQL • OpenAI
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_insight_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in normalized.split("\n")]

    compacted_lines = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        compacted_lines.append(line)
        previous_blank = is_blank

    return "\n".join(compacted_lines).strip()


def is_bullet_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("- ", "* ", "• "))


def strip_bullet_marker(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith(("• ", "- ", "* ")):
        return stripped[2:].strip()
    return stripped


def render_business_insight(answer: str) -> None:
    normalized = normalize_insight_text(answer)
    if not normalized:
        st.info("No business insight was generated.")
        return

    html_parts: list[str] = ['<div class="insight-report">']
    bullet_buffer: list[str] = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            paragraph = " ".join(part.strip() for part in paragraph_buffer if part.strip())
            if paragraph:
                html_parts.append(f"<p>{escape(paragraph)}</p>")
            paragraph_buffer.clear()

    def flush_bullets() -> None:
        if bullet_buffer:
            html_parts.append("<ul>")
            for bullet in bullet_buffer:
                html_parts.append(f"<li>{escape(bullet)}</li>")
            html_parts.append("</ul>")
            bullet_buffer.clear()

    for line in normalized.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            flush_paragraph()
            flush_bullets()
            continue

        if is_bullet_line(stripped_line):
            flush_paragraph()
            bullet_buffer.append(strip_bullet_marker(stripped_line))
            continue

        flush_bullets()
        paragraph_buffer.append(stripped_line)

    flush_paragraph()
    flush_bullets()
    html_parts.append("</div>")

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def render_agent_result(result: dict[str, Any], tracker: dict[str, Any] | None = None) -> None:
    errors = result.get("errors", [])
    answer = result.get("answer", "")

    if errors:
        st.error("The agent could not complete the request.")
        with st.expander("Error details"):
            for error in errors:
                st.write(error)
        render_reasoning_pipeline(
            result,
            make_visualization_stage("skipped", {"reason": "Agent workflow failed before visualization."}),
        )
        if tracker:
            update_progress_tracker(tracker, failed_stage="Rendering charts")
        return

    st.subheader("💡 Business Insight")
    render_business_insight(answer)
    render_execution_metadata(result)

    render_generated_sql(result)

    dataframe = build_results_dataframe(result)
    render_results_table(dataframe, result)

    if tracker:
        update_progress_tracker(tracker, active_stage="Rendering charts")

    visualization_stage = render_automatic_charts(dataframe)

    if tracker:
        if visualization_stage.get("status") == "failed":
            update_progress_tracker(tracker, failed_stage="Rendering charts")
        else:
            update_progress_tracker(tracker, completed_stages=("Rendering charts",))

    render_reasoning_pipeline(result, visualization_stage)

    with st.expander("Analysis details"):
        selected_tables = result.get("selected_tables", [])
        st.write("Selected tables:", selected_tables)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-logo">⚡ {PAGE_TITLE}</div>
                <div class="sidebar-subtitle">{PAGE_SUBTITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-title">🤖 Workflow Status</div>',
            unsafe_allow_html=True,
        )
        for stage in LOADING_STAGES:
            st.caption(f"○ {stage}")

        st.markdown(
            '<div class="sidebar-section-title">🕘 Analytics History</div>',
            unsafe_allow_html=True,
        )
        history = st.session_state.get(QUERY_HISTORY_KEY, [])
        if not history:
            st.caption("No questions submitted yet.")
            return

        for index, item in enumerate(history, start=1):
            status = "Failed" if item["has_errors"] else f"{item['row_count']} rows"
            with st.expander(f"{index}. {item['question'][:60]}", expanded=False):
                st.write(status)
                if item["sql_query"]:
                    st.code(item["sql_query"], language="sql")


def main() -> None:
    st.cache_resource.clear()
    render_page_header()
    initialize_session_state()
    render_kpi_cards()

    question = render_question_form()
    
    if question:
        st.session_state["current_question"] = question
        tracker = render_progress_tracker()
        try:
            result = run_agent_with_progress(question, tracker)
            st.session_state["current_result"] = result
            add_query_history(question, result)
        except Exception as exc:
            logger.exception("Retail analytics frontend request failed.")
            st.error("An unexpected error occurred while processing the question.")
            with st.expander("Error details"):
                st.write(str(exc))
            render_sidebar()
            render_footer()
            return

    render_sidebar()
    
    if "current_result" in st.session_state:
        render_agent_result(st.session_state["current_result"])

    render_footer()


if __name__ == "__main__":
    main()
