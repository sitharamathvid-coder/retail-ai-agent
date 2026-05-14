from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from security.prompt_guard import is_prompt_safe
from agent.rag import get_rag_context

load_dotenv(override=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

STAGE_PROMPT_VALIDATED = "Prompt security validation"
STAGE_QUESTION_RECEIVED = "Question received"
STAGE_TABLES_SELECTED = "Tables selected"
STAGE_SCHEMAS_ANALYZED = "Schemas analyzed"
STAGE_SQL_GENERATED = "SQL generated"
STAGE_QUERY_VALIDATED = "Query validated"
STAGE_QUERY_EXECUTED = "Query executed"
STAGE_INTENT_CLARIFIED = "Intent clarified"
STAGE_INSIGHTS_GENERATED = "Insights generated"


class RetailAgentState(TypedDict):
    question: str
    tables: list[str]
    selected_tables: list[str]
    schemas: dict[str, list[dict[str, str]]]
    sql_query: str
    query_result: dict[str, Any]
    answer: str
    errors: list[str]
    sql_errors: list[str]
    clarification_message: str | None
    retry_count: int
    metadata: dict[str, Any]
    workflow_stages: list[dict[str, Any]]


def get_mcp_server_url() -> str:
    return os.getenv("MCP_SERVER_URL", "http://localhost:8000").rstrip("/")


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    kwargs = {"model": model_name, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
        
    return ChatOpenAI(**kwargs)


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    return json.loads(cleaned, strict=False)


def call_mcp_get(path: str) -> dict[str, Any]:
    url = f"{get_mcp_server_url()}{path}"
    logger.info("Calling MCP GET %s", path)
    with httpx.Client(timeout=20) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def call_mcp_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{get_mcp_server_url()}{path}"
    logger.info("Calling MCP POST %s", path)
    with httpx.Client(timeout=60) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def mark_stage(
    state: RetailAgentState,
    name: str,
    status: str,
    details: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> RetailAgentState:
    stage = {
        "name": name,
        "status": status,
        "details": details or {},
    }
    if elapsed_seconds is not None:
        stage["elapsed_seconds"] = round(elapsed_seconds, 3)

    return {**state, "workflow_stages": [*state["workflow_stages"], stage]}


def fail_stage(
    state: RetailAgentState,
    name: str,
    error: str,
    elapsed_seconds: float | None = None,
) -> RetailAgentState:
    return mark_stage(
        state,
        name,
        "failed",
        {"error": error},
        elapsed_seconds,
    )


def validate_user_prompt(state: RetailAgentState) -> RetailAgentState:
    started_at = time.perf_counter()
    is_safe, reason = is_prompt_safe(state["question"])
    
    if not is_safe:
        answer = "Unsafe or unauthorized instructions were detected in the request. This analytics environment only supports governed read-only business intelligence queries."
        next_state = {**state, "errors": [*state["errors"], reason], "answer": answer}
        return fail_stage(next_state, "Unsafe prompt blocked", reason, time.perf_counter() - started_at)
        
    return mark_stage(
        state,
        STAGE_PROMPT_VALIDATED,
        "completed",
        {"status": "safe"},
        time.perf_counter() - started_at,
    )


def clarify_query_intent(state: RetailAgentState) -> RetailAgentState:
    if state["errors"]:
        return state

    started_at = time.perf_counter()
    
    prompt = f"""
You are a query intent analyst for a retail analytics system.
Your goal is to determine if the user's question is specific enough to be answered by a SQL query against a retail database.

Question:
{state["question"]}

Guidelines:
- Do NOT request clarification for standard aggregate ranking queries (e.g., "top sellers", "highest revenue", "top categories", "best states", "Which sellers generate the most revenue?", "Top payment methods"). For these, automatically default to the full available dataset with no timeframe or regional filtering and return `needs_clarification: false`.
- ONLY trigger clarification if:
  1. The metric is completely undefined.
  2. There are conflicting dimensions.
  3. Required temporal granularity is missing for trend analysis (e.g., "Show growth trends", "Compare recent performance", "Which products are best?", "Which sellers improved?").
  4. A safe SQL query cannot be logically inferred.
- If clarification is needed, return `needs_clarification: true` and provide a helpful `clarification_message`.
- If the question is a greeting or unrelated, return `needs_clarification: true` and a polite response.

Return JSON only:
{{
  "needs_clarification": boolean,
  "clarification_message": "string or null"
}}
"""

    try:
        response = get_llm().invoke(prompt)
        parsed = parse_json_object(str(response.content))
        
        needs_clarification = parsed.get("needs_clarification", False)
        clarification_message = parsed.get("clarification_message")
        
        if needs_clarification:
            logger.info("Query requires clarification: %s", clarification_message)
            return mark_stage(
                {**state, "clarification_message": clarification_message, "answer": clarification_message},
                STAGE_INTENT_CLARIFIED,
                "needs_clarification",
                {"message": clarification_message},
                time.perf_counter() - started_at
            )
            
        return mark_stage(
            {**state, "clarification_message": None},
            STAGE_INTENT_CLARIFIED,
            "completed",
            {"status": "clear"},
            time.perf_counter() - started_at
        )
    except Exception as exc:
        logger.error("Intent clarification failed: %s", exc)
        # Fallback to proceeding anyway if clarification check fails
        return state


def list_available_tables(state: RetailAgentState) -> RetailAgentState:
    if state["errors"]:
        return state

    started_at = time.perf_counter()
    
    try:
        payload = call_mcp_get("/tables")
        tables = payload.get("tables", [])
        logger.info("Discovered %s PostgreSQL tables.", len(tables))
        metadata = {**state["metadata"], "available_table_count": len(tables)}
        return {**state, "tables": tables, "metadata": metadata}
    except (httpx.HTTPError, ValueError) as exc:
        error = f"Unable to retrieve database tables: {exc}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        return fail_stage(next_state, STAGE_TABLES_SELECTED, error, time.perf_counter() - started_at)


def choose_relevant_tables(state: RetailAgentState) -> RetailAgentState:
    if state["errors"] or not state["tables"]:
        return state

    started_at = time.perf_counter()
    prompt = f"""
You are planning database tool usage for a retail analytics agent.

Question:
{state["question"]}

Available PostgreSQL tables:
{json.dumps(state["tables"], indent=2)}

Return JSON only with this shape:
{{
  "selected_tables": ["table_name"]
}}

Select only tables needed to answer the question. If unsure, choose the most relevant retail tables.
"""

    try:
        response = get_llm().invoke(prompt)
        parsed = parse_json_object(str(response.content))
        selected_tables = [
            table
            for table in parsed.get("selected_tables", [])
            if table in state["tables"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Table selection failed, falling back to all available tables: %s", exc)
        selected_tables = state["tables"]
    except Exception as exc:
        error = f"OpenAI table selection failed: {exc}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        return fail_stage(next_state, STAGE_TABLES_SELECTED, error, time.perf_counter() - started_at)

    if not selected_tables:
        selected_tables = state["tables"]

    logger.info("Selected tables for schema inspection: %s", ", ".join(selected_tables))
    metadata = {**state["metadata"], "selected_tables": selected_tables}
    next_state = {**state, "selected_tables": selected_tables, "metadata": metadata}
    return mark_stage(
        next_state,
        STAGE_TABLES_SELECTED,
        "completed",
        {"selected_tables": selected_tables},
        time.perf_counter() - started_at,
    )


def fetch_selected_schemas(state: RetailAgentState) -> RetailAgentState:
    if state["errors"] or not state["selected_tables"]:
        return state

    started_at = time.perf_counter()
    schemas: dict[str, list[dict[str, str]]] = {}
    errors = list(state["errors"])

    for table_name in state["selected_tables"]:
        try:
            payload = call_mcp_get(f"/schema/{table_name}")
            schemas[table_name] = payload.get("columns", [])
        except (httpx.HTTPError, ValueError) as exc:
            error = f"Unable to retrieve schema for table '{table_name}': {exc}"
            logger.exception(error)
            errors.append(error)

    logger.info("Loaded schemas for %s tables.", len(schemas))
    next_state = {**state, "schemas": schemas, "errors": errors}
    status = "failed" if errors else "completed"
    return mark_stage(
        next_state,
        STAGE_SCHEMAS_ANALYZED,
        status,
        {"schema_table_count": len(schemas), "errors": errors},
        time.perf_counter() - started_at,
    )


def generate_sql_query(state: RetailAgentState) -> RetailAgentState:
    if state["errors"] or not state["schemas"]:
        return state

    started_at = time.perf_counter()
    reflection_prompt = ""
    if state.get("sql_errors"):
        reflection_prompt = f"""
IMPORTANT - PREVIOUS QUERY FAILED:
Your previous attempts to execute SQL generated the following errors:
{json.dumps(state['sql_errors'], indent=2)}

Please analyze the error and provide a CORRECTED SQL query. Pay close attention to data types, table joins, and column names.
"""

    rag_context = get_rag_context(state["question"])

    prompt = f"""
You are a senior analytics engineer generating PostgreSQL for a retail analytics question.

Business Logic & Schema Rules:
{rag_context}

Rules:
- Return JSON only.
- Generate exactly one read-only SELECT query.
- Do not use DELETE, DROP, TRUNCATE, ALTER, UPDATE, INSERT, CREATE, GRANT, REVOKE, CALL, EXECUTE, or MERGE.
- Use only the tables and columns provided in the schema.
- Prefer concise aggregations that answer the business question.
- Include readable aliases for business metrics.
- IMPORTANT: If the user asks for multiple metrics (e.g., both 'Revenue' and 'Order Count'), ensure BOTH metrics are included in the SELECT clause with clear aliases (e.g., total_revenue and order_count).
- CRITICAL ALIAS RULE: Small models often hallucinate aliases. DO NOT use short table aliases. You MUST use the full table name for all column references (e.g., `customers.customer_state` instead of `c.state`) to prevent "missing FROM-clause entry" errors.
- GLOBAL REFLECTION: If the Business Rules or your reasoning require a table that is NOT present in the 'Available schemas', do NOT write SQL. Instead, list the required tables in `missing_tables` and leave `sql_query` empty.
{reflection_prompt}
Question:
{state["question"]}

Available schemas:
{json.dumps(state["schemas"], indent=2)}

Return JSON with this shape:
{{
  "sql_query": "select ...",
  "missing_tables": ["table1"] // Optional. Only if you cannot proceed without more tables.
}}
"""

    try:
        response = get_llm().invoke(prompt)
        parsed = parse_json_object(str(response.content))
        
        missing_tables_requested = parsed.get("missing_tables", [])
        missing_tables = [t for t in missing_tables_requested if t in state["tables"] and t not in state["selected_tables"]]
        
        if missing_tables:
            logger.warning("Global Reflection triggered: LLM requested missing tables: %s", missing_tables)
            next_state = {
                **state, 
                "selected_tables": list(set(state["selected_tables"] + missing_tables)),
                "sql_query": ""
            }
            return mark_stage(
                next_state,
                STAGE_SQL_GENERATED,
                "completed",
                {"global_reflection": True, "missing_tables_added": missing_tables},
                time.perf_counter() - started_at,
            )
            
        sql_query = str(parsed.get("sql_query", "")).strip()
        if not sql_query:
            raise ValueError("LLM returned empty SQL query but did not request valid missing tables.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        error = f"Unable to generate SQL query: {exc}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        return fail_stage(next_state, STAGE_SQL_GENERATED, error, time.perf_counter() - started_at)
    except Exception as exc:
        error = f"OpenAI SQL generation failed: {exc}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        return fail_stage(next_state, STAGE_SQL_GENERATED, error, time.perf_counter() - started_at)

    logger.info("Generated SQL query: %s", sql_query)
    metadata = {**state["metadata"], "generated_sql": sql_query}
    next_state = {**state, "sql_query": sql_query, "metadata": metadata}
    return mark_stage(
        next_state,
        STAGE_SQL_GENERATED,
        "completed",
        {"sql_query": sql_query},
        time.perf_counter() - started_at,
    )


def execute_sql_query(state: RetailAgentState) -> RetailAgentState:
    if state["errors"] or not state["sql_query"]:
        return state

    started_at = time.perf_counter()
    try:
        payload = call_mcp_post("/query", {"query": state["sql_query"], "limit": 100})
        row_count = payload.get("row_count", 0)
        logger.info("MCP query returned %s rows.", row_count)
        metadata = {**state["metadata"], "rows_returned": row_count}
        next_state = {**state, "query_result": payload, "metadata": metadata}
        next_state = mark_stage(
            next_state,
            STAGE_QUERY_VALIDATED,
            "completed",
            {"validator": "mcp_server.POST /query"},
            None,
        )
        return mark_stage(
            next_state,
            STAGE_QUERY_EXECUTED,
            "completed",
            {"rows_returned": row_count},
            time.perf_counter() - started_at,
        )
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
            
        if exc.response.status_code == 400 and state.get("retry_count", 0) < 3:
            logger.warning("SQL execution failed. Retrying... Error: %s", detail)
            next_state = {
                **state,
                "retry_count": state.get("retry_count", 0) + 1,
                "sql_errors": [*state.get("sql_errors", []), detail],
            }
            return next_state
            
        error = f"MCP query execution rejected or failed after retries: {detail}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        next_state = fail_stage(
            next_state,
            STAGE_QUERY_VALIDATED,
            error,
            time.perf_counter() - started_at,
        )
        return fail_stage(next_state, STAGE_QUERY_EXECUTED, error, None)
    except (httpx.HTTPError, ValueError) as exc:
        error = f"Unable to execute SQL query through MCP server: {exc}"
        logger.exception(error)
        next_state = {**state, "errors": [*state["errors"], error]}
        next_state = fail_stage(
            next_state,
            STAGE_QUERY_VALIDATED,
            error,
            time.perf_counter() - started_at,
        )
        return fail_stage(next_state, STAGE_QUERY_EXECUTED, error, None)


def generate_business_insights(state: RetailAgentState) -> RetailAgentState:
    if state["errors"]:
        answer = (
            "I could not complete the retail analytics request because one or more "
            f"tool calls failed: {'; '.join(state['errors'])}"
        )
        return fail_stage({**state, "answer": answer}, STAGE_INSIGHTS_GENERATED, answer, None)

    started_at = time.perf_counter()
    
    try:
        import pandas as pd
        from agent.insight_generator import generate_executive_insight
        
        rows = state["query_result"].get("rows", [])
        if rows:
            df = pd.DataFrame(rows)
            answer = generate_executive_insight(df, state["question"], state["sql_query"])
        else:
            answer = "The SQL query returned no results, so no business insights can be generated."
            
    except Exception as exc:
        error = f"OpenAI insight generation failed: {exc}"
        logger.exception(error)
        answer = f"The SQL query completed, but the final insight generation failed: {exc}"
        next_state = {**state, "answer": answer, "errors": [*state["errors"], error]}
        return fail_stage(
            next_state,
            STAGE_INSIGHTS_GENERATED,
            error,
            time.perf_counter() - started_at,
        )

    total_execution_time = time.perf_counter() - state["metadata"].get(
        "started_at",
        time.perf_counter(),
    )
    metadata = {
        **state["metadata"],
        "execution_time_seconds": round(total_execution_time, 3),
    }
    next_state = {**state, "answer": answer, "metadata": metadata}
    return mark_stage(
        next_state,
        STAGE_INSIGHTS_GENERATED,
        "completed",
        {"answer_length": len(answer)},
        time.perf_counter() - started_at,
    )


def build_retail_agent():
    graph = StateGraph(RetailAgentState)
    graph.add_node("validate_user_prompt", validate_user_prompt)
    graph.add_node("clarify_query_intent", clarify_query_intent)
    graph.add_node("list_available_tables", list_available_tables)
    graph.add_node("choose_relevant_tables", choose_relevant_tables)
    graph.add_node("fetch_selected_schemas", fetch_selected_schemas)
    graph.add_node("generate_sql_query", generate_sql_query)
    graph.add_node("execute_sql_query", execute_sql_query)
    graph.add_node("generate_business_insights", generate_business_insights)

    graph.set_entry_point("validate_user_prompt")
    
    def route_after_validation(state: RetailAgentState) -> str:
        if state["errors"]:
            return END
        return "clarify_query_intent"
        
    graph.add_conditional_edges("validate_user_prompt", route_after_validation)
    
    def route_after_intent(state: RetailAgentState) -> str:
        if state.get("clarification_message"):
            return END
        return "list_available_tables"
        
    graph.add_conditional_edges("clarify_query_intent", route_after_intent)
    
    graph.add_edge("list_available_tables", "choose_relevant_tables")
    graph.add_edge("choose_relevant_tables", "fetch_selected_schemas")
    graph.add_edge("fetch_selected_schemas", "generate_sql_query")
    
    def route_after_sql_generation(state: RetailAgentState) -> str:
        if state["errors"]:
            return END
        if not state.get("sql_query") and state.get("selected_tables"):
            return "fetch_selected_schemas"
        return "execute_sql_query"

    graph.add_conditional_edges("generate_sql_query", route_after_sql_generation)
    
    def route_after_execution(state: RetailAgentState) -> str:
        if state["errors"]:
            return END
        if state.get("sql_errors") and state.get("retry_count", 0) > 0 and not state.get("query_result"):
            return "generate_sql_query"
        return "generate_business_insights"

    graph.add_conditional_edges("execute_sql_query", route_after_execution)
    graph.add_edge("generate_business_insights", END)
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retail analytics LangGraph agent.")
    parser.add_argument("question", help="Natural language retail analytics question to answer.")
    return parser.parse_args()


def create_initial_state(question: str) -> RetailAgentState:
    started_at = time.perf_counter()
    return {
        "question": question,
        "tables": [],
        "selected_tables": [],
        "schemas": {},
        "sql_query": "",
        "query_result": {},
        "answer": "",
        "errors": [],
        "sql_errors": [],
        "clarification_message": None,
        "retry_count": 0,
        "metadata": {
            "started_at": started_at,
            "question": question,
        },
        "workflow_stages": [
            {
                "name": STAGE_QUESTION_RECEIVED,
                "status": "completed",
                "details": {"question": question},
                "elapsed_seconds": 0,
            }
        ],
    }


def main() -> int:
    args = parse_args()
    agent = build_retail_agent()
    result = agent.invoke(create_initial_state(args.question))
    print(result["answer"])
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
