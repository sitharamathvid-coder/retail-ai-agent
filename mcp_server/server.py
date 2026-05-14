from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://retail_user:retail_password@localhost:5432/retail_analytics"
)
BLOCKED_SQL_PATTERN = re.compile(
    r"\b(delete|drop|update|insert|alter|truncate|create|grant|revoke)\b",
    re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
if DATABASE_URL == DEFAULT_DATABASE_URL:
    logger.warning("DATABASE_URL is not set. Using local development default.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(
    title="Retail AI Analytics MCP Server",
    description="Service boundary for retail analytics tools exposed to AI agents.",
    version="0.1.0",
)


class AnalyticsQuery(BaseModel):
    question: str = Field(..., min_length=3, description="Business question to answer.")
    limit: int = Field(default=20, ge=1, le=100)


class SqlQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Read-only SELECT SQL query.")
    limit: int = Field(default=100, ge=1, le=1000)


class ToolResult(BaseModel):
    tool: str
    result: dict[str, Any]


def reject_query(reason: str, query: str) -> None:
    compact_query = " ".join(query.split())
    logger.warning("Rejected SQL query. Reason: %s | Query: %s", reason, compact_query)
    raise HTTPException(status_code=400, detail=reason)


def strip_comments_and_literals(query: str) -> str:
    without_comments = re.sub(
        r"--.*?$|/\*.*?\*/",
        " ",
        query,
        flags=re.MULTILINE | re.DOTALL,
    )
    without_single_quoted_strings = re.sub(
        r"'(?:''|[^'])*'",
        " ",
        without_comments,
    )
    without_double_quoted_identifiers = re.sub(
        r'"(?:""|[^"])*"',
        " ",
        without_single_quoted_strings,
    )
    return without_double_quoted_identifiers


def validate_read_only_query(query: str) -> str:
    normalized_query = query.strip()

    if not normalized_query:
        reject_query("SQL query cannot be empty.", query)

    statement = normalized_query.rstrip(";").strip()
    validation_target = strip_comments_and_literals(statement)

    if not validation_target.strip():
        reject_query("SQL query cannot be empty after removing comments and literals.", query)

    if ";" in validation_target:
        reject_query("Only a single SELECT statement is allowed.", query)

    first_token = re.match(r"^\s*([a-zA-Z]+)\b", validation_target)
    if not first_token or first_token.group(1).lower() != "select":
        reject_query("Only read-only SELECT queries are allowed.", query)

    blocked_match = BLOCKED_SQL_PATTERN.search(validation_target)
    if blocked_match:
        blocked_keyword = blocked_match.group(1).upper()
        reject_query(f"SQL query contains blocked keyword: {blocked_keyword}.", query)

    if "customer_pii" in validation_target.lower():
        reject_query("Security Violation: Access to customer_pii is forbidden.", query)

    logger.info("SQL query passed read-only validation.")

    return statement


def fetch_tables() -> list[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                select table_name
                from information_schema.tables
                where table_schema = 'public'
                  and table_type = 'BASE TABLE'
                order by table_name
                """
            )
        ).mappings()
        return [row["table_name"] for row in rows]


def fetch_table_schema(table_name: str) -> list[dict[str, str]]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                select column_name, data_type
                from information_schema.columns
                where table_schema = 'public'
                  and table_name = :table_name
                order by ordinal_position
                """
            ),
            {"table_name": table_name},
        ).mappings()
        return [
            {"column_name": row["column_name"], "data_type": row["data_type"]}
            for row in rows
        ]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "retail-ai-analytics-mcp"}


@app.get("/tools")
def list_tools() -> dict[str, list[dict[str, str]]]:
    return {
        "tools": [
            {
                "name": "database_health",
                "description": "Validate PostgreSQL connectivity.",
            },
            {
                "name": "top_tables",
                "description": "List public schema tables available for analytics.",
            },
            {
                "name": "schema",
                "description": "Inspect public schema table columns and PostgreSQL data types.",
            },
            {
                "name": "query",
                "description": "Execute guarded read-only SELECT queries.",
            },
        ]
    }


@app.get("/tables")
def list_tables() -> dict[str, list[str]]:
    logger.info("Listing PostgreSQL tables.")
    try:
        tables = fetch_tables()
    except SQLAlchemyError as exc:
        logger.exception("Unable to list PostgreSQL tables.")
        raise HTTPException(status_code=503, detail="Unable to list database tables.") from exc

    return {"tables": tables}


@app.get("/schema/{table_name}")
def get_table_schema(table_name: str) -> dict[str, Any]:
    logger.info("Fetching schema for table '%s'.", table_name)
    try:
        columns = fetch_table_schema(table_name)
    except SQLAlchemyError as exc:
        logger.exception("Unable to fetch schema for table '%s'.", table_name)
        raise HTTPException(status_code=503, detail="Unable to fetch table schema.") from exc

    if not columns:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

    return {"table_name": table_name, "columns": columns}


@app.post("/query")
def execute_query(request: SqlQueryRequest) -> dict[str, Any]:
    statement = validate_read_only_query(request.query)
    limited_statement = f"select * from ({statement}) as readonly_query limit :limit"
    logger.info("Executing read-only query with limit %s.", request.limit)

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(limited_statement),
                {"limit": request.limit},
            ).mappings()
            records = [dict(row) for row in rows]
    except SQLAlchemyError as exc:
        logger.exception("Read-only query execution failed.")
        error_msg = str(exc.orig) if hasattr(exc, "orig") else str(exc)
        raise HTTPException(status_code=400, detail=f"Query execution failed: {error_msg}") from exc

    return {"row_count": len(records), "rows": records}


@app.post("/tools/database_health", response_model=ToolResult)
def database_health() -> ToolResult:
    logger.info("Checking database health.")
    try:
        with engine.connect() as connection:
            database_time = connection.execute(text("select now()")).scalar_one()
    except SQLAlchemyError as exc:
        logger.exception("Database health check failed.")
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc

    return ToolResult(
        tool="database_health",
        result={"status": "ok", "database_time": str(database_time)},
    )


@app.post("/tools/top_tables", response_model=ToolResult)
def top_tables(query: AnalyticsQuery) -> ToolResult:
    logger.info("Listing top tables for question: %s", query.question)
    try:
        tables = fetch_tables()[: query.limit]
    except SQLAlchemyError as exc:
        logger.exception("Unable to list tables for tool request.")
        raise HTTPException(status_code=503, detail="Unable to list tables.") from exc

    return ToolResult(tool="top_tables", result={"question": query.question, "tables": tables})
