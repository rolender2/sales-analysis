"""
Main entry point for the Multi-Agent Sales Data Analysis System.

Usage:
    python main.py --run-pipeline          Run full analysis pipeline
    python main.py --run-agent <name>      Run a specific agent
    python main.py --list-agents           List available agents
    python main.py --check-setup           Verify configuration
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from agents import Runner, trace, set_tracing_export_api_key
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    MONGODB_URI,
    TRACE_WORKFLOW_NAME,
    validate_api_keys,
    get_model_string,
    LLM_PROVIDERS,
)


def check_setup():
    """Verify system configuration."""
    print("=" * 60)
    print("Multi-Agent Sales Data Analysis System - Setup Check")
    print("=" * 60)
    
    # Check API keys
    print("\n📋 API Key Status:")
    api_status = validate_api_keys()
    for provider, configured in api_status.items():
        status = "✅ Configured" if configured else "❌ Not set"
        print(f"  {provider.capitalize()}: {status}")
    
    # Check MongoDB connection
    print("\n🗄️  MongoDB Connection:")
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print(f"  ✅ Connected to {MONGODB_URI}")
        
        # Check collections
        db = client["superstore"]
        collections = db.list_collection_names()
        print(f"  📁 Collections: {', '.join(collections)}")
        
        if "sales" in collections:
            count = db.sales.count_documents({})
            print(f"  📊 Sales records: {count:,}")
        
        client.close()
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
    
    # Check output directories
    print("\n📂 Output Directories:")
    from config import REPORTS_DIR, VISUALIZATIONS_DIR
    print(f"  Reports: {REPORTS_DIR} {'✅' if REPORTS_DIR.exists() else '❌'}")
    print(f"  Visualizations: {VISUALIZATIONS_DIR} {'✅' if VISUALIZATIONS_DIR.exists() else '❌'}")
    
    print("\n" + "=" * 60)


def list_agents():
    """List available agents."""
    print("\n📋 Available Agents:")
    print("-" * 40)
    
    agents_info = [
        ("data_reviewer", "DataReviewer", "Profiles and assesses data quality"),
        ("data_cleaner", "DataCleaner", "Cleans and transforms raw data"),
        ("data_analyst", "DataAnalyst", "Performs comprehensive analysis"),
        ("forecaster", "Forecaster", "Generates sales forecasts"),
        ("conversational", "ConversationalAgent", "Natural language queries"),
        ("orchestrator", "Orchestrator", "Coordinates the pipeline"),
    ]
    
    for agent_id, name, description in agents_info:
        print(f"  • {agent_id:18} - {description}")
    
    print()


async def run_single_agent(agent_name: str, provider: str = "openai", model: str = None):
    """Run a single agent."""
    print(f"\n🚀 Running {agent_name} agent...")
    print("-" * 40)
    
    # Import and create the agent
    if agent_name == "data_reviewer":
        from sales_agents.data_reviewer import create_data_reviewer
        agent = create_data_reviewer(provider, model)
        prompt = "Profile the superstore.sales collection and generate a comprehensive data quality report."
    
    elif agent_name == "data_cleaner":
        from sales_agents.data_cleaner import create_data_cleaner
        agent = create_data_cleaner(provider, model)
        prompt = "Clean the superstore.sales data and create the sales_cleaned collection."
    
    elif agent_name == "data_analyst":
        from sales_agents.data_analyst import create_data_analyst
        agent = create_data_analyst(provider, model)
        prompt = "Analyze the sales_cleaned collection and generate comprehensive analysis with visualizations."
    
    elif agent_name == "forecaster":
        from sales_agents.forecaster import create_forecaster
        agent = create_forecaster(provider, model)
        prompt = "Generate sales forecasts using multiple models and create the forecast report."
    
    else:
        print(f"❌ Unknown agent: {agent_name}")
        return
    
    # Enable tracing
    if OPENAI_API_KEY:
        set_tracing_export_api_key(OPENAI_API_KEY)
    
    # Run the agent with tracing
    with trace(f"{agent_name.replace('_', ' ').title()} Execution") as t:
        print(f"📍 Trace ID: {t.trace_id}")
        print(f"🔗 View trace: https://platform.openai.com/traces/{t.trace_id}")
        print()
        
        result = await Runner.run(agent, prompt)
        
        print("\n✅ Agent completed!")
        print("-" * 40)
        print(f"Output:\n{result.final_output[:2000]}...")


async def run_pipeline(provider: str = "openai", model: str = None):
    """Run the full analysis pipeline with isolated contexts per agent."""
    import uuid
    
    print("\n" + "=" * 60)
    print("🚀 Starting Sales Data Analysis Pipeline")
    print("=" * 60)
    
    # Enable tracing
    if OPENAI_API_KEY:
        set_tracing_export_api_key(OPENAI_API_KEY)
    
    from sales_agents.data_reviewer import create_data_reviewer
    from sales_agents.data_cleaner import create_data_cleaner
    from sales_agents.data_analyst import create_data_analyst
    from sales_agents.forecaster import create_forecaster
    
    # Create unique group ID for this pipeline
    pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
    print(f"\n📍 Pipeline Group ID: {pipeline_id}")
    print(f"🔗 View traces: https://platform.openai.com/traces?group_id={pipeline_id}")
    
    # Stage 1: Data Review (fresh context)
    print("\n" + "-" * 40)
    print("📊 Stage 1: Data Review")
    print("-" * 40)
    reviewer = create_data_reviewer(provider, model)
    with trace("1. Data Reviewer", group_id=pipeline_id):
        review_result = await Runner.run(
            reviewer, 
            "Profile the superstore.sales collection.",
            max_turns=30
        )
    print("✅ Data Review complete")
    
    # Stage 2: Data Cleaning (fresh context)
    print("\n" + "-" * 40)
    print("🧹 Stage 2: Data Cleaning")
    print("-" * 40)
    cleaner = create_data_cleaner(provider, model)
    with trace("2. Data Cleaner", group_id=pipeline_id):
        clean_result = await Runner.run(
            cleaner,
            "Clean the sales data and create sales_cleaned.",
            max_turns=30
        )
    print("✅ Data Cleaning complete")
    
    # Stage 3: Data Analysis (fresh context)
    print("\n" + "-" * 40)
    print("📈 Stage 3: Data Analysis")
    print("-" * 40)
    analyst = create_data_analyst(provider, model)
    with trace("3. Data Analyst", group_id=pipeline_id):
        analysis_result = await Runner.run(
            analyst,
            "Analyze the sales data with visualizations.",
            max_turns=60
        )
    print("✅ Data Analysis complete")
    
    # Stage 4: Forecasting (fresh context)
    print("\n" + "-" * 40)
    print("🔮 Stage 4: Forecasting")
    print("-" * 40)
    forecaster_agent = create_forecaster(provider, model)
    with trace("4. Forecaster", group_id=pipeline_id):
        forecast_result = await Runner.run(
            forecaster_agent,
            "Generate sales forecasts.",
            max_turns=40
        )
    print("✅ Forecasting complete")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\n📍 View traces: https://platform.openai.com/traces?group_id={pipeline_id}")
    print("\n📁 Generated outputs:")
    print("  • outputs/reports/data_review_report.md")
    print("  • outputs/reports/data_cleaning_log.md")
    print("  • outputs/reports/analysis.md")
    print("  • outputs/reports/forecast_report.md")
    print("  • outputs/visualizations/*.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Sales Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full analysis pipeline",
    )
    
    parser.add_argument(
        "--run-agent",
        type=str,
        metavar="AGENT",
        help="Run a specific agent (data_reviewer, data_cleaner, data_analyst, forecaster)",
    )
    
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agents",
    )
    
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Verify system configuration",
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "deepseek"],
        help="LLM provider to use (default: openai)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use (default: provider's default)",
    )
    
    args = parser.parse_args()
    
    if args.check_setup:
        check_setup()
    elif args.list_agents:
        list_agents()
    elif args.run_agent:
        asyncio.run(run_single_agent(args.run_agent, args.provider, args.model))
    elif args.run_pipeline:
        asyncio.run(run_pipeline(args.provider, args.model))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
