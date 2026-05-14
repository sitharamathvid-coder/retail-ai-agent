import os
import logging

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

BUSINESS_RULES = [
    "To calculate Revenue, you must sum the `payment_value` column in the `payments` table.",
    "If answering geographic questions (like state, city, or zip code) using the `geolocation` table, you MUST join it to the `customers` table via `customer_zip_code_prefix`.",
    "The `orders` table joins to the `customers` table via `customer_id`.",
    "The `orders` table joins to the `order_items` table via `order_id`.",
    "The `order_items` table joins to the `products` table via `product_id`.",
    "EXAMPLE SQL for state revenue: SELECT customer_state, SUM(payment_value) as total_revenue FROM orders JOIN customers ON orders.customer_id = customers.customer_id JOIN payments ON orders.order_id = payments.order_id GROUP BY customer_state",
    "CRITICAL: Do not use any table aliases (like 'c' or 'p'). Always use raw column names."
]

_vector_store = None

def get_embeddings() -> OpenAIEmbeddings:
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    kwargs = {"model": model_name}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
        
    return OpenAIEmbeddings(**kwargs, check_embedding_ctx_length=False)


def get_rag_context(question: str) -> str:
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing InMemoryVectorStore for RAG rules.")

        embeddings = get_embeddings()
        _vector_store = InMemoryVectorStore.from_texts(BUSINESS_RULES, embedding=embeddings)
    
    docs = _vector_store.similarity_search(question, k=3)
    return "\n".join([f"- {doc.page_content}" for doc in docs])
