"""
Data Cleaner Agent - Agent 2

This agent cleans and transforms the raw sales data,
creating a new sales_cleaned collection.
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
    fix_date_formats,
)
from config import get_model_string, MONGODB_DATABASE

# =============================================================================
# Data Cleaner Agent Definition
# =============================================================================

DATA_CLEANER_INSTRUCTIONS = """Analyze the `sales` collection for data quality issues and create a cleaning proposal.

## CRITICAL: Avoid Context Overflow
- NEVER query more than 100 records at once.
- Use `limit` parameter in all queries.
- Use aggregation pipelines with $sample, $match, and $limit.
- For defect detection, query specific conditions (e.g., {State: {$regex: "typo"}}) rather than scanning all.

## 1. Detect & Fix Defects

### Step 1: Bulk Date Fix (Do This FIRST)
Call `fix_date_formats("sales", "Order Date")` immediately. This handles 5,000+ date issues efficiently.

### Step 2: Targeted Defect Queries (Use LIMIT 100)
- **Typos**: Query `{State: {$regex: "_typo|Californa|Ohhio", $options: "i"}, limit: 100}`.
- **Nulls**: Query `{$or: [{Region: null}, {Category: null}, {Segment: null}], limit: 100}`.
- **Outliers**: Query `{Sales: {$gt: 50000}, limit: 100}` to find extreme values.
- **Duplicates**: Use aggregation: `[{$group: {_id: {fields...}, count: {$sum: 1}}}, {$match: {count: {$gt: 1}}}, {$limit: 50}]`.

## 2. Generate Proposal
MANDATORY: You MUST create BOTH files:
1. `cleaning_proposal.csv` - Contains all defect records with columns: RecordID, Field, CurrentValue, ProposedValue, Reason
2. `cleaning_proposal.md` - Summary of issues found

### A. CSV Proposal (`cleaning_proposal.csv`)
Columns: `RecordID`, `Field`, `CurrentValue`, `ProposedValue`, `Reason`
- Maximum 200 rows in the CSV (if more defects exist, note in markdown).

### B. Markdown Summary (`cleaning_proposal.md`)
- Summary of issues found.
- Table showing count of proposed changes by type.
- Sample of 5 records from the proposal.

FINAL STEPS (MANDATORY):
1. Call `save_report("cleaning_proposal.csv", <csv_content>)`
2. Call `save_report("cleaning_proposal.md", <markdown_content>)`
"""


def create_data_cleaner(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Data Cleaner agent with the specified LLM provider.
    
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
        name="DataCleaner",
        model=model_string,
        instructions=DATA_CLEANER_INSTRUCTIONS,
        tools=[query_mongodb, save_report, get_collection_stats, fix_date_formats],
    )


# Default instance using OpenAI
data_cleaner = create_data_cleaner()
