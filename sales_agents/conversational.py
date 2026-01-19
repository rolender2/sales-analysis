"""
Conversational Query Agent - Agent 5

This agent handles natural language queries about the sales data,
generating MongoDB queries and providing conversational responses.
"""

from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent

from sales_agents.tools import (
    query_mongodb,
    save_visualization,
    create_dataframe_visualization,
    get_collection_stats,
)
from config import get_model_string, MONGODB_DATABASE

# =============================================================================
# Conversational Agent Definition
# =============================================================================

CONVERSATIONAL_INSTRUCTIONS = """You are an expert Sales Data Analyst.
Dont just give numbers—give business context.

## Data Source
Query the `sales` collection (~9,800 records).
Fields: "Order ID", "Category", "Sub-Category", "Region", "Segment", "Sales" (string), "Profit" (float), "Discount" (float).

## Response Style
1. **Analyst Persona**: e.g., "While Furniture has high volume ($X), it suffers from low margins due to high discounts."
2. **Comparisons**: Always try to compare vs the average. "This is 20% higher than the monthly average."
3. **Pro-active**: If they ask for Sales, check Profit too. High sales means nothing if we lost money.

## Technical Rules
- Use `query_mongodb` with aggregation pipelines.
- Sales is a STRING -> `{"$toDouble": "$Sales"}`.
- Profit/Discount are numbers.
- Format currency as $1,234.56.

## Example
User: "How did tables do?"
You: "Tables generated **$124k in revenue**, making them the 4th largest sub-category. However, they are **unprofitable (-$17k)** due to an aggressive average discount of 26%. This is a key area for margin improvement."
"""


def create_conversational_agent(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Conversational Query agent with the specified LLM provider.
    
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
        name="ConversationalAgent",
        model=model_string,
        instructions=CONVERSATIONAL_INSTRUCTIONS,
        tools=[
            query_mongodb,
            save_visualization,
            create_dataframe_visualization,
            get_collection_stats,
        ],
    )


# Default instance using OpenAI
conversational_agent = create_conversational_agent()
