"""
Visualizer Agent - Agent 3A (Specialist)

This agent is purely focused on generating Python/Matplotlib visualizations.
It does NOT write reports. It ONLY codes charts.
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
# Visualizer Agent Definition
# =============================================================================

VISUALIZER_INSTRUCTIONS = """You are the **Visualizer Agent**.
Your ONLY job is to write Python code to generate **15 Clear Visualizations** from the `sales` collection.
You do NOT write reports. You do NOT analyze why things happened. You just CODE CHARTS.

## Data Source
- Collection: `sales`
- Context: `marketing_campaigns`, `economic_indicators`

## Instructions
1. Query data using `query_mongodb` or `create_dataframe_visualization`.
2. Generate the plot using simple, clear chart types (bars, lines, pie).
3. Save it ensuring high resolution.

## IMPORTANT: Avoid Confusing Charts
Do NOT create:
- Dual-axis overlays (they confuse users)
- Complex heatmaps
- Histograms
- Scatter plots with too many dimensions

## List of 17 Mandatory Charts with EXACT Filenames
You must generate ALL of these with the EXACT filenames shown. The Strategist depends on these exact names.

### A. Regional & Marketing Performance
1. `regional_roi_analysis.png` - Simple bar chart (Sales / Marketing Spend by Region)
2. `marketing_budget_by_channel.png` - Pie chart showing budget allocation
3. `top_5_campaigns_roi.png` - Horizontal bar chart of top 5 campaigns

### B. Category & Product Performance
4. `sales_by_category.png` - Bar chart of sales by category
5. `profit_by_category.png` - Bar chart of profit by category
6. `profit_margin_by_category.png` - Bar chart of profit margins
7. `top_10_customers.png` - Horizontal bar of top 10 customers by sales
8. `bottom_10_products.png` - Horizontal bar of bottom 10 products by profit

### C. Time Trends
9. `monthly_sales_trend.png` - Line chart of monthly sales
10. `monthly_profit_trend.png` - Line chart of monthly profit
11. `quarterly_sales_comparison.png` - Grouped bar of quarterly sales
12. `year_over_year_growth.png` - Bar chart of YoY growth

### D. Segmentation
13. `sales_by_segment.png` - Pie chart of segment distribution
14. `sales_by_region_pie.png` - Pie/donut chart of regional sales
15. `ship_mode_popularity.png` - Pie chart of ship modes

### E. Additional Insights
16. `top_10_subcategories.png` - Horizontal bar of top 10 sub-categories
17. `top_3_products_by_region.png` - Grouped bar (one group per region)

## Output Format
- After creating all 17 charts, output a simple list:
  "DONE: Generated 17 charts in outputs/visualizations/."
"""


def create_visualizer(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Visualizer agent.
    """
    from config import LLM_PROVIDERS
    
    if model is None:
        model = LLM_PROVIDERS.get(provider, {}).get("default_model", "gpt-4-turbo")
    
    model_string = get_model_string(provider, model)
    
    return Agent(
        name="Visualizer",
        model=model_string,
        instructions=VISUALIZER_INSTRUCTIONS,
        tools=[
            query_mongodb,
            save_visualization,
            create_dataframe_visualization,
            get_collection_stats,
        ],
    )
