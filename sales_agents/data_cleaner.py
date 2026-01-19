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
)
from config import get_model_string, MONGODB_DATABASE

# =============================================================================
# Data Cleaner Agent Definition
# =============================================================================

DATA_CLEANER_INSTRUCTIONS = """Analyze the `sales` collection for data quality issues and create a cleaning proposal.

## Workflow
1. **Analyze**: Use `query_mongodb` and `get_collection_stats` to find issues (e.g., string 'Sales' that should be numbers, inconsistent variations of 'Region', null values).
2. **Propose**: Generate a CSV file (`cleaning_proposal.csv`) containing record-level changes.
3. **Summarize**: Generate a Markdown report (`cleaning_proposal.md`) summarizing the findings.

## Output Format
### 1. CSV Proposal (`cleaning_proposal.csv`)
Columns: `RecordID`, `Field`, `CurrentValue`, `ProposedValue`, `Reason`
- **RecordID**: The `_id` of the document.
- **Field**: The field name to update.
- **Reason**: Why the change is needed.

Use `save_report("cleaning_proposal.csv", csv_content, "text")` to save this file.
*Example*:
`RecordID,Field,CurrentValue,ProposedValue,Reason`
`507f1f77bcf86cd799439011,Sales,"1,200.00",1200.00,"Convert string to number"`

### 2. Markdown Summary (`cleaning_proposal.md`)
- Summary of issues found.
- Table showing count of proposed changes by type.
- Sample of 5 records from the proposal.
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
        tools=[query_mongodb, save_report, get_collection_stats],
    )


# Default instance using OpenAI
data_cleaner = create_data_cleaner()
