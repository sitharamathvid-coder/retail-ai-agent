import json
import logging
import re
from typing import Tuple

import pandas as pd

from agent.insight_engine import (
    clean_column_name,
    generate_statistical_context,
)

logger = logging.getLogger(__name__)

def summarize_top_entities(dataframe: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Pre-processes the SQL result dataframe to mask raw UUIDs.
    """
    df = dataframe.copy()
    summary = []
    
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    
    for col in df.columns:
        if df[col].dtype == object and len(df) > 0:
            sample_val = str(df[col].iloc[0])
            if uuid_pattern.match(sample_val) or (len(sample_val) >= 32 and "id" in col.lower()):
                entity_name = clean_column_name(col).replace(" Id", "")
                df[col] = [f"Top {entity_name} #{i+1}" for i in range(len(df))]
                summary.append(f"- Masked {col} to 'Top {entity_name} #N'.")

    return "\n".join(summary), df

def generate_executive_insight(dataframe: pd.DataFrame, question: str, sql_query: str) -> str:
    """Generates an executive-grade business insight narrative using the LLM with statistical context."""
    from agent.agent import get_llm
    
    logger.info("Executive insight generation started.")
    
    if dataframe.empty:
        return "The query returned no data. No insights can be generated."

    masking_summary, masked_df = summarize_top_entities(dataframe)
    
    stat_context = generate_statistical_context(masked_df)

    data_sample = masked_df.head(10).to_dict(orient="records")
    
    prompt = f"""
You are an elite BI strategy consultant. Provide a highly polished, Tableau Pulse/Power BI Copilot-style executive narrative.

OUTPUT FORMAT EXACTLY AS SHOWN BELOW:
💡 Executive Summary
[Dynamic 1-sentence headline summarizing the core strategic finding]

📊 Key Findings
• [Primary KPI insight with exact numbers]
• [Concentration or comparative insight using the Computed Statistical Context below]
• [Additional relevant business finding based on the data shape]

⚠️ Risks / Opportunities
• [Identified anomaly, dependency risk (mention severity if available), or growth opportunity]

📈 Recommendation
• [Actionable, strategic executive recommendation]

RULES:
- Be concise, direct, and authoritative. NO filler like "Based on the query results".
- Incorporate the computed statistical context (percentages, gaps, severities) to ground the narrative in math.
- Never expose raw UUIDs.
- Format all money as localized currency (e.g., R$ 229.5K).
- Maximum 5 bullet points total.

Question asked: {question}

{stat_context}

Data Sample (masked for security):
{json.dumps(data_sample, indent=2, default=str)}

Generate the executive narrative now:
"""

    try:
        response = get_llm(temperature=0.1).invoke(prompt)
        answer = str(response.content).strip()
        logger.info("Executive BI narrative generated successfully.")
        return answer
    except Exception as exc:
        logger.exception("Executive insight generation failed: %s", exc)
        return f"Failed to generate executive insight: {exc}"
