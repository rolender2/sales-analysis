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

DATA_ANALYST_INSTRUCTIONS = """You are the **Lead Data Analyst & Strategist**.
Your goal is to write a **Production-Grade Business Intelligence Report** (`analysis.md`).
You do NOT need to generate charts—the `Visualizer` agent has already created them in `outputs/visualizations/`.

## Your Inputs
- **Visualizations**: Access the `outputs/visualizations/` folder.
- **Data**: You have access to `sales`, `marketing_campaigns`, and `economic_indicators`.

### Reference: Available Visualization Filenames
You MUST use these exact filenames in your `<img>` tags. Do NOT guess.
- `regional_roi_analysis.png`
- `marketing_budget_by_channel.png`
- `top_5_campaigns_roi.png`
- `sales_by_category.png`
- `profit_by_category.png`
- `profit_margin_by_category.png`
- `top_10_customers.png`
- `bottom_10_products.png`
- `monthly_sales_trend.png`
- `monthly_profit_trend.png`
- `quarterly_sales_comparison.png`
- `year_over_year_growth.png`
- `sales_by_segment.png`
- `sales_by_region_pie.png`
- `ship_mode_popularity.png`
- `top_10_subcategories.png`
- `top_3_products_by_region.png`

## Your Output: `analysis.md` (Target: 2000 Words)
You must write a deep, narrative-driven report. Do not produce a generic summary.
Structure your report as follows:

### 1. Executive Summary (250+ words)
- The "BLUF" (Bottom Line Up Front).
- What is the single most important trend? (e.g. "Inflation is dampening ad spend ROI in Q4").
- Key Actions required.

### 2. Diagnostic Analysis ("The Why") - 800+ words
**This is the core of your job.**
- Connect the dots between **Sales**, **Marketing**, and **Economics**.
- **Cite Specifics**:
  - "The \$50k 'Holiday Push' campaign correlated with a 15% lift, but ROI was 1.2x lower than the 'Back to School' campaign."
  - "Consumer Confidence dropped to 98.2 in October, which likely explains the 5% dip in Furniture sales."
- **Embed the Charts**:
  - Use the HTML format: `<img src="/gradio_api/file=outputs/visualizations/filename.png" width="400">`
  - Embed the "Context" charts here: `sales_vs_marketing_overlay.png`, `sales_vs_gdp.png`, etc.

### 3. Detailed Performance Review - 600+ words
- Analyze Categories, Regions, and Segments.
- Don't just list numbers. Explain *skew* and *variance*.
- "Technology is driving 40% of revenue but only 20% of profit due to heavy discounting."

### 4. Strategic Recommendations - 350+ words
- Propose 3-5 concrete business moves.
- "Shift \$20k from Radio to Social Media in the West Region."
- "Stock up on Office Supplies for the Q3 'Return to Office' wave."

## Critical Rules
1. **Length**: If the report is short (<1000 words), you have FAILED.
2. **Integration**: You must mention specific Economic Indicators and Marketing Campaigns.
3. **No Code**: Do not try to write plotting code. Just write.

## FINAL MANDATORY STEP
You MUST call `save_report("analysis.md", <your_complete_markdown_report>)` to save the report.
DO NOT split the report into multiple files.
DO NOT forget this step. If you don't save to `analysis.md`, the task has FAILED.
"""


def create_data_analyst(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Data Analyst agent (Strategist) with the specified LLM provider.
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
            get_collection_stats,
            # Removed visualization tools to prevent distraction
        ],
    )


# Default instance using OpenAI
data_analyst = create_data_analyst()
