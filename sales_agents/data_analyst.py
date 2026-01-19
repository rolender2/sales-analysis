"""
Data Analyst Agent - Agent 3

This agent performs comprehensive sales analysis on the cleaned data
and generates visualizations.
"""

from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent

from sales_agents.tools import (
    query_mongodb,
    save_report,
    save_visualization,
    create_dataframe_visualization,
    get_collection_stats,
)
from config import get_model_string, MONGODB_DATABASE

# =============================================================================
# Data Analyst Agent Definition
# =============================================================================

DATA_ANALYST_INSTRUCTIONS = """You are a Senior Data Analyst.
Analyze the `sales` collection (~9,800 records) efficiently using aggregation pipelines.
Never fetch raw records or use the `sales_cleaned` collection.

## 1. Explore & Analyze (Batch Mode)
- **Profitability**: `[{"$group": {"_id": "$Category", "profit": {"$sum": "$Profit"}, "sales": {"$sum": {"$toDouble": "$Sales"}}}}]`
- **Segments**: Compare avg order value by Segment.

## 2. Visualize (Iterative)
You MUST generate at least **18 distinct visualizations** to fully explore the data.
Use `save_visualization` for each. You can batch 2-3 plots per turn if possible, or run sequentially.
IMPORTANT: When embedding images in the report, ALWAYS use the following HTML format to control size:
`<img src="/gradio_api/file=outputs/visualizations/filename.png" alt="Description" width="300">`
Examples of REQUIRED Charts:
1.  **Sales by Category** (Bar)
2.  **Profit by Category** (Bar)
3.  **Sales vs Profit Scatter** (Scatter)
4.  **Sales by Region** (Pie/Donut)
5.  **Profit Map by Region** (Bar or Heatmap)
6.  **Monthly Sales Trend** (Line)
7.  **Monthly Profit Trend** (Line)
8.  **Sales Distribution** (Histogram)
9.  **Top 10 Customers by Sales** (Bar)
10. **Bottom 10 Profitable Products** (Bar)
11. **Discount Impact on Profit** (Scatter)
12. **Sales by Segment** (Bar)
13. **Profit Margin by Category** (Bar)
14. **Ship Mode Popularity** (Pie)
15. **Sales by State** (Top 10)
16. **Profit Ratio by Sub-Category** (Bar)
17. **Order Quantity Distribution** (Box Plot)
18. **Weekday Sales Analysis** (Heatmap/Bar)

## 3. Report Findings (analysis.md)
Write a detailed, professional `analysis.md` report (approx. 1500 words).
Structure it as a formal business presentation:
- **Executive Summary**: The "So What?" of the analysis.
- **Methodology**: Briefly explain how you analyzed the data.
- **Detailed Findings**: Dedicated section for EACH of the 18+ visualizations.
    - Embed the image using the HTML format with `width="60%"`.
    - **Robust Analysis**: Write 3-4 distinct paragraphs per observation.
    - Include specific numbers, calculating percentage changes where relevant.
    - Discuss outliers, trends, and business implications in depth.
    - Avoid generic statements like "Sales are good."
- **Strategic Recommendations**: Provide concrete, actionable business advice based on the data.
"""


def create_data_analyst(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Data Analyst agent with the specified LLM provider.
    
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
        name="DataAnalyst",
        model=model_string,
        instructions=DATA_ANALYST_INSTRUCTIONS,
        tools=[
            query_mongodb,
            save_report,
            save_visualization,
            create_dataframe_visualization,
            get_collection_stats,
        ],
    )


# Default instance using OpenAI
data_analyst = create_data_analyst()
