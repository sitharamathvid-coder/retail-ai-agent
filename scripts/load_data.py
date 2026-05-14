from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://retail_user:retail_password@localhost:5432/retail_analytics"
)

CSV_TABLE_MAPPING = {
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_customers_dataset.csv": "customers",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "category_translation",
}

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL is not set. Using local development default.")
        return DEFAULT_DATABASE_URL

    return database_url


def load_csv_to_postgres(csv_path: Path, table_name: str, engine) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    dataframe.to_sql(table_name, engine, if_exists="replace", index=False)
    return len(dataframe)


def load_mapped_csv_files(data_dir: Path) -> dict[str, int]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    database_url = get_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)
    results: dict[str, int] = {}
    failures: list[str] = []

    for file_name, table_name in CSV_TABLE_MAPPING.items():
        csv_path = data_dir / file_name

        if not csv_path.exists():
            logger.warning("Skipping missing file: %s", csv_path)
            continue

        logger.info("Loading %s into table '%s'.", csv_path, table_name)

        try:
            rows_loaded = load_csv_to_postgres(csv_path, table_name, engine)
        except (OSError, pd.errors.ParserError, SQLAlchemyError, ValueError) as exc:
            failures.append(file_name)
            logger.exception("Failed to load %s into table '%s': %s", csv_path, table_name, exc)
            continue

        results[table_name] = rows_loaded
        logger.info("Loaded %s rows into table '%s'.", rows_loaded, table_name)

    if failures:
        failed_files = ", ".join(failures)
        raise RuntimeError(f"Failed to load one or more CSV files: {failed_files}")

    if not results:
        logger.warning("No mapped CSV files were loaded from %s.", data_dir)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load mapped Olist retail CSV files from the data folder into PostgreSQL."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the mapped CSV files. Defaults to ./data.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        results = load_mapped_csv_files(args.data_dir)
    except Exception as exc:
        logger.error("Data load failed: %s", exc)
        return 1

    total_rows = sum(results.values())
    logger.info("Data load completed. Loaded %s total rows across %s tables.", total_rows, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
