"""
Orchestrator Agent - Agent 6

This agent coordinates the multi-agent pipeline using handoffs
and generates the executive summary.
"""

from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent

from sales_agents.tools import save_report, get_collection_stats
from config import get_model_string, MONGODB_DATABASE

# Import other agents for handoffs
from sales_agents.data_reviewer import data_reviewer
from sales_agents.data_cleaner import data_cleaner
from sales_agents.data_analyst import data_analyst
from sales_agents.forecaster import forecaster

# =============================================================================
# Orchestrator Agent Definition
# =============================================================================

ORCHESTRATOR_INSTRUCTIONS = """You are the Pipeline Orchestrator for the Sales Data Analysis System.

Your role is to coordinate the execution of specialized agents in the correct order
and ensure the full analysis pipeline completes successfully.

## Pipeline Stages:

1. **Data Review** (DataReviewer agent)
   - Profiles the raw superstore.sales data
   - Generates data_review_report.md
   - Flags data quality issues

2. **Data Cleaning** (DataCleaner agent)
   - Cleans and transforms the data
   - Creates sales_cleaned collection
   - Generates data_cleaning_log.md

3. **Data Analysis** (DataAnalyst agent)
   - Performs comprehensive sales analysis
   - Creates visualizations
   - Generates analysis.md

4. **Forecasting** (Forecaster agent)
   - Generates sales forecasts
   - Creates forecast visualizations
   - Generates forecast_report.md

## Coordination Protocol:

When you receive a request to run the pipeline:

1. **Handoff to DataReviewer**
   - Say "Starting data review phase..."
   - Transfer control to review the raw data
   - Wait for completion

2. **Handoff to DataCleaner**
   - Say "Data review complete. Starting cleaning phase..."
   - Transfer control to clean the data
   - Wait for completion

3. **STOP - User Approval Required**
   - **CRITICAL**: Do NOT proceed to DataAnalyst or Forecaster.
   - Say: "Data cleaning proposal generated. Please review 'cleaning_proposal.csv' and apply changes in the UI before continuing."
   - **Terminate execution** here. The user must manually trigger the next stages (Analysis/Forecasting) after they are satisfied with the data.

## Important - Resume Protocol:
If the user explicitly asks to "Run Analysis" or "Run Forecasting" (implying data is clean), ONLY THEN proceed to those agents.

4. **Handoff to DataAnalyst** (Only if explicitly requested)
   - Say "Starting analysis phase..."
   - Transfer control for analysis
   - Wait for completion

5. **Handoff to Forecaster** (Only if explicitly requested)
   - Say "Starting forecasting phase..."
   - Transfer control for forecasting
   - Wait for completion

## Handoff Usage:

To hand off to another agent, simply indicate you want to transfer control:
"Handing off to DataReviewer to profile the sales data."

The framework will automatically route to the appropriate agent.

## Error Handling:

If an agent encounters an error:
1. Log the error
2. Attempt to continue with remaining stages if possible
3. Note any skipped stages in the executive summary

## Important Notes:
- Always provide status updates between stages
- Verify each stage completed successfully before moving on
- The executive summary should be comprehensive but concise
- Reference specific values from each agent's output
"""


def create_orchestrator(provider: str = "openai", model: str = None) -> Agent:
    """
    Create an Orchestrator agent with the specified LLM provider.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'deepseek')
        model: Specific model name (uses provider default if not specified)
        
    Returns:
        Configured Agent instance with handoffs to other agents
    """
    from config import LLM_PROVIDERS
    
    if model is None:
        model = LLM_PROVIDERS.get(provider, {}).get("default_model", "gpt-4-turbo")
    
    model_string = get_model_string(provider, model)
    
    return Agent(
        name="Orchestrator",
        model=model_string,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        tools=[save_report, get_collection_stats],
        handoffs=[data_reviewer, data_cleaner, data_analyst, forecaster],
    )


# Default instance using OpenAI
orchestrator = create_orchestrator()
