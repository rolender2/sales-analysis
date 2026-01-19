"""
Data Reviewer Agent - Agent 1

This agent profiles the superstore.sales collection and generates
a comprehensive data quality report.
"""

from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent

from sales_agents.tools import (
    query_mongodb,
    save_report,
    get_collection_stats,
)
from config import get_model_string, MONGODB_DATABASE

# =============================================================================
# Data Reviewer Agent Definition
# =============================================================================

DATA_REVIEWER_INSTRUCTIONS = """Act as a Forensic Data Auditor.
Profile `superstore.sales` efficiently using AGGREGATION PIPELINES.
DO NOT inspect records one-by-one.

## 1. Structural Audit (Use Pipelines)
- **Mixed Types**: Check if any 'Sales' are not strings.
  `[{"$match": {"Sales": {"$not": {"$type": "string"}}}}, {"$count": "bad_types"}]`
- **Date Consistency**: Check min/max len of Order Date.

## 2. Value Distribution (Use Pipelines)
- **Negative Values**: Count negative profits.
  `[{"$match": {"Profit": {"$lt": 0}}}, {"$count": "losses"}]`
- **Outliers**: Get Max Sales.
  `[{"$addFields": {"s": {"$toDouble": "$Sales"}}}, {"$group": {"_id": null, "max_sales": {"$max": "$s"}}}]`

## 3. Generate Report (data_review_report.md)
Save a comprehensive, professional `data_review_report.md`. It must be detailed (approx. 500 words).
Include independent sections for:
- **Executive Summary**: High-level health check of the data.
- **Structural Integrity**: Detailed audit of types, nulls, and schema consistency.
- **Value Distribution**: Deep dive into the numbers (ranges, averages, negative values).
- **Anomalies & Outliers**: Specific list of suspicious records found.
- **Cleaning Strategy**: Detailed recommended actions for the cleaning agent.
"""


def create_data_reviewer(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Data Reviewer agent with the specified LLM provider.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'deepseek')
        model: Specific model name (uses provider default if not specified)
        
    Returns:
        Configured Agent instance
    """
    from config import LLM_PROVIDERS
    
    if model is None:
        model = LLM_PROVIDERS.get(provider, {}).get("default_model", "gpt-4-turbo")
    
    model_string = get_model_string(provider, model)
    
    return Agent(
        name="DataReviewer",
        model=model_string,
        instructions=DATA_REVIEWER_INSTRUCTIONS,
        tools=[query_mongodb, save_report, get_collection_stats],
    )


# Default instance using OpenAI
data_reviewer = create_data_reviewer()
