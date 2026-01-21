"""
Gradio Frontend for Multi-Agent Sales Data Analysis System.

A multi-tab interface for:
- Dashboard with KPIs
- Data Review reports
- Data Cleaning logs
- Analysis with visualizations
- Forecasting results
- Conversational Query interface
- Settings

Usage:
    python app.py
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from agents import Runner, trace, set_tracing_export_api_key

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    OUTPUTS_DIR,
    LLM_PROVIDERS,
    get_model_string,
    validate_api_keys,
    MONGODB_URI,
    MONGODB_DATABASE,
)

# =============================================================================
# Helper Functions
# =============================================================================

def get_available_providers():
    """Get list of providers with configured API keys."""
    api_status = validate_api_keys()
    return [p for p, configured in api_status.items() if configured]


def get_models_for_provider(provider: str):
    """Get available models for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("models", [])


def load_report(filename: str) -> str:
    """Load a report file content. Supports split files (e.g., report_part2.md)."""
    filepath = REPORTS_DIR / filename
    full_content = []
    
    # Load main file if it exists
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            full_content.append(f.read())
    
    # Always check for additional part files (e.g., forecast_report_part2.md)
    if filename.endswith(".md"):
        base_name = filename[:-3]
        pattern = f"{base_name}_part*.md"
        parts = sorted(list(REPORTS_DIR.glob(pattern)))
        
        for part in parts:
            with open(part, "r", encoding="utf-8") as f:
                full_content.append(f.read())
    
    if full_content:
        return "\n\n---\n\n".join(full_content)
            
    return f"*Report not found: {filename}*\n\nRun the pipeline to generate reports."


def get_visualizations():
    """Get list of visualization files."""
    if VISUALIZATIONS_DIR.exists():
        return list(VISUALIZATIONS_DIR.glob("*.png"))
    return []


def get_dashboard_stats():
    """Get dashboard statistics from MongoDB."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        
        stats = {}
        
        # Always use the full sales collection
        collection = db.sales
        stats["collection"] = "sales"
        
        # Total records
        stats["total_records"] = collection.count_documents({})
        
        # Total sales - convert string to double
        pipeline = [
            {"$addFields": {"s": {"$toDouble": "$Sales"}}},
            {"$group": {"_id": None, "total": {"$sum": "$s"}}}
        ]
        result = list(collection.aggregate(pipeline))
        stats["total_sales"] = result[0]["total"] if result else 0
        
        # Unique customers
        pipeline = [{"$group": {"_id": "$Customer ID"}}, {"$count": "count"}]
        result = list(collection.aggregate(pipeline))
        stats["unique_customers"] = result[0]["count"] if result else 0
        
        # Categories
        pipeline = [{"$group": {"_id": "$Category"}}, {"$count": "count"}]
        result = list(collection.aggregate(pipeline))
        stats["categories"] = result[0]["count"] if result else 0
        
        client.close()
        return stats

    # ... (existing code) ...

        viz_gallery = gr.Gallery(
            value=get_forecast_viz(),
            label="Forecast Charts",
            columns=2,
            height="auto",
            interactive=False,  # Fixes the "Drop Media" issue
        )
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Agent Running Functions
# =============================================================================

async def run_agent_async(agent_name: str, prompt: str, provider: str, model: str):
    """Run an agent asynchronously."""
    from sales_agents.data_reviewer import create_data_reviewer
    from sales_agents.data_cleaner import create_data_cleaner
    from sales_agents.data_analyst import create_data_analyst
    from sales_agents.forecaster import create_forecaster
    from sales_agents.conversational import create_conversational_agent
    
    # Create agent based on name
    if agent_name == "data_reviewer":
        agent = create_data_reviewer(provider, model)
    elif agent_name == "data_cleaner":
        agent = create_data_cleaner(provider, model)
    elif agent_name == "data_analyst":
        agent = create_data_analyst(provider, model)
    elif agent_name == "forecaster":
        agent = create_forecaster(provider, model)
    elif agent_name == "conversational":
        agent = create_conversational_agent(provider, model)
    else:
        return f"Unknown agent: {agent_name}"
    
    # Enable tracing
    if OPENAI_API_KEY:
        set_tracing_export_api_key(OPENAI_API_KEY)
    
    # Run with trace
    trace_name = f"{agent_name.replace('_', ' ').title()} - {datetime.now().strftime('%H:%M:%S')}"
    with trace(trace_name) as t:
        result = await Runner.run(agent, prompt)
        trace_url = f"https://platform.openai.com/traces/{t.trace_id}"
    
    return result.final_output, trace_url


def run_pipeline_phase1(provider: str, model: str, progress=gr.Progress()):
    """Run Phase 1 of pipeline: Data Review and Data Cleaning.
    Halts after cleaning for user approval before proceeding to analysis."""
    
    async def _run():
        from agents import Runner
        from agents.tracing import trace, set_tracing_export_api_key
        from sales_agents.data_reviewer import create_data_reviewer
        from sales_agents.data_cleaner import create_data_cleaner
        
        if OPENAI_API_KEY:
            set_tracing_export_api_key(OPENAI_API_KEY)
        
        results = []
        
        # Create a unique group ID for this pipeline run
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        trace_url = None
        
        # Stage 1: Data Reviewer (fresh context)
        progress(0.2, desc="Running Data Reviewer...")
        reviewer = create_data_reviewer(provider, model)
        with trace("1. Data Reviewer", group_id=pipeline_id) as t1:
            if trace_url is None:
                trace_url = f"https://platform.openai.com/traces?group_id={pipeline_id}"
            r1 = await Runner.run(reviewer, "Profile the superstore.sales collection.", max_turns=30)
        results.append(("Data Review", "✅"))
        
        # Stage 2: Data Cleaner (fresh context)
        progress(0.6, desc="Running Data Cleaner...")
        cleaner = create_data_cleaner(provider, model)
        with trace("2. Data Cleaner", group_id=pipeline_id) as t2:
            r2 = await Runner.run(cleaner, "Clean the sales data and create sales_cleaned.", max_turns=60)
        results.append(("Data Cleaning", "✅"))
        
        progress(1.0, desc="Phase 1 Complete - Review cleaning proposal!")
        results.append(("⏸️ HALTED", "Review proposal, then click 'Apply Changes' to fix data and generate Data Quality Certificate"))
        return results, trace_url, pipeline_id
    
    return asyncio.run(_run())


def run_pipeline_phase2(provider: str, model: str, pipeline_id: str = None, progress=gr.Progress()):
    """Run Phase 2 of pipeline: Data Analysis and Forecasting.
    Should only be run after user approves the cleaning proposal."""
    
    async def _run():
        from agents import Runner
        from agents.tracing import trace, set_tracing_export_api_key
        from sales_agents.data_analyst import create_data_analyst
        from sales_agents.forecaster import create_forecaster
        
        if OPENAI_API_KEY:
            set_tracing_export_api_key(OPENAI_API_KEY)
        
        results = []
        
        # Use existing pipeline ID or create new one
        if not pipeline_id:
            pid = f"pipeline_{uuid.uuid4().hex[:8]}"
        else:
            pid = pipeline_id
        trace_url = f"https://platform.openai.com/traces?group_id={pid}"
        
        # Stage 3A: Visualizer (fresh context)
        progress(0.2, desc="Running Visualizer (Generating Charts)...")
        from sales_agents.visualizer import create_visualizer
        visualizer = create_visualizer(provider, model)
        with trace("3A. Visualizer", group_id=pid) as t3a:
            r3a = await Runner.run(visualizer, "Generate 22 visualizations for the sales analysis.", max_turns=80)
        results.append(("Visualizations", "✅"))

        # Stage 3B: Data Analyst aka Strategist (fresh context)
        progress(0.5, desc="Running Strategist (Writing Report)...")
        analyst = create_data_analyst(provider, model)
        with trace("3B. Strategist", group_id=pid) as t3b:
            r3b = await Runner.run(analyst, "Write the 2000-word analysis.md report based on the visualizations.", max_turns=100)
        results.append(("Analysis", "✅"))
        
        # Stage 4: Forecaster (fresh context)
        progress(0.8, desc="Running Forecaster...")
        forecaster = create_forecaster(provider, model)
        with trace("4. Forecaster", group_id=pid) as t4:
            r4 = await Runner.run(forecaster, "Generate sales forecasts.", max_turns=80)
        results.append(("Forecasting", "✅"))
        
        progress(1.0, desc="Pipeline complete!")
        return results, trace_url
    
    return asyncio.run(_run())


def run_full_pipeline(provider: str, model: str, progress=gr.Progress()):
    """Run the full pipeline without halting (legacy behavior)."""
    
    async def _run():
        from agents import Runner
        from agents.tracing import trace, set_tracing_export_api_key
        from sales_agents.data_reviewer import create_data_reviewer
        from sales_agents.data_cleaner import create_data_cleaner
        from sales_agents.data_analyst import create_data_analyst
        from sales_agents.forecaster import create_forecaster
        
        if OPENAI_API_KEY:
            set_tracing_export_api_key(OPENAI_API_KEY)
        
        results = []
        
        # Create a unique group ID for this pipeline run
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        trace_url = None
        
        # Stage 1: Data Reviewer
        progress(0.1, desc="Running Data Reviewer...")
        reviewer = create_data_reviewer(provider, model)
        with trace("1. Data Reviewer", group_id=pipeline_id) as t1:
            if trace_url is None:
                trace_url = f"https://platform.openai.com/traces?group_id={pipeline_id}"
            r1 = await Runner.run(reviewer, "Profile the superstore.sales collection.", max_turns=30)
        results.append(("Data Review", "✅"))
        
        # Stage 2: Data Cleaner
        progress(0.3, desc="Running Data Cleaner...")
        cleaner = create_data_cleaner(provider, model)
        with trace("2. Data Cleaner", group_id=pipeline_id) as t2:
            r2 = await Runner.run(cleaner, "Clean the sales data and create sales_cleaned.", max_turns=30)
        results.append(("Data Cleaning", "✅"))
        
        # Stage 3: Data Analyst
        progress(0.6, desc="Running Data Analyst...")
        analyst = create_data_analyst(provider, model)
        with trace("3. Data Analyst", group_id=pipeline_id) as t3:
            r3 = await Runner.run(analyst, "Analyze the sales data with visualizations.", max_turns=60)
        results.append(("Analysis", "✅"))
        
        # Stage 4: Forecaster
        progress(0.9, desc="Running Forecaster...")
        forecaster = create_forecaster(provider, model)
        with trace("4. Forecaster", group_id=pipeline_id) as t4:
            r4 = await Runner.run(forecaster, "Generate sales forecasts.", max_turns=40)
        results.append(("Forecasting", "✅"))
        
        progress(1.0, desc="Pipeline complete!")
        return results, trace_url
    
    return asyncio.run(_run())


# =============================================================================
# Gradio Interface Components
# =============================================================================

def create_dashboard_tab():
    """Create the Dashboard tab."""
    with gr.Tab("Dashboard"):
        gr.Markdown("# 📊 Sales Analytics Dashboard")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
        
        with gr.Row():
            total_records = gr.Number(label="Total Records", interactive=False)
            total_sales = gr.Number(label="Total Sales ($)", interactive=False)
            unique_customers = gr.Number(label="Unique Customers", interactive=False)
            categories = gr.Number(label="Categories", interactive=False)
        
        with gr.Row():
            collection_info = gr.Textbox(label="Data Source", interactive=False)
        
        def refresh_stats():
            stats = get_dashboard_stats()
            if "error" in stats:
                return 0, 0, 0, 0, f"Error: {stats['error']}"
            return (
                stats.get("total_records", 0),
                stats.get("total_sales", 0),
                stats.get("unique_customers", 0),
                stats.get("categories", 0),
                f"Collection: {stats.get('collection', 'N/A')}",
            )
        
        refresh_btn.click(
            refresh_stats,
            outputs=[total_records, total_sales, unique_customers, categories, collection_info],
        )
        
        gr.Markdown("---")
        gr.Markdown("### 🚀 Run Analysis Pipeline")
        gr.Markdown("*Phase 1 runs Review + Clean, then halts for approval. Phase 2 runs Analysis + Forecasting.*")
        
        with gr.Row():
            pipeline_provider = gr.Dropdown(
                choices=list(LLM_PROVIDERS.keys()),
                value="deepseek",
                label="LLM Provider",
            )
            pipeline_model = gr.Dropdown(
                choices=get_models_for_provider("deepseek"),
                value="deepseek-chat",
                label="Model",
            )
        
        pipeline_id_state = gr.State(value=None)
        
        with gr.Row():
            run_phase1_btn = gr.Button("1️⃣ Run Phase 1 (Review + Clean)", variant="primary")
            run_phase2_btn = gr.Button("2️⃣ Run Phase 2 (Analyze + Forecast)", variant="secondary")
        
        with gr.Row():
            run_full_btn = gr.Button("⚡ Run Full Pipeline (No Halt)", variant="secondary", size="sm")
        
        pipeline_status = gr.Textbox(label="Pipeline Status", lines=6, interactive=False)
        trace_link = gr.Markdown("")
        
        def update_models(provider):
            models = get_models_for_provider(provider)
            default = models[0] if models else ""
            return gr.Dropdown(choices=models, value=default)
        
        pipeline_provider.change(update_models, inputs=[pipeline_provider], outputs=[pipeline_model])
        
        def execute_phase1(provider, model, progress=gr.Progress()):
            try:
                results, trace_url, pid = run_pipeline_phase1(provider, model, progress)
                status = "\n".join([f"{stage}: {result}" for stage, result in results])
                status += "\n\n📋 Review the cleaning_proposal.md, then click Phase 2 to continue."
                link = f"[🔗 View Trace]({trace_url})"
                return status, link, pid
            except Exception as e:
                return f"❌ Error: {str(e)}", "", None
        
        def execute_phase2(provider, model, pid, progress=gr.Progress()):
            try:
                results, trace_url = run_pipeline_phase2(provider, model, pid, progress)
                status = "\n".join([f"✅ {stage}: {result}" for stage, result in results])
                link = f"[🔗 View Trace]({trace_url})"
                return status, link
            except Exception as e:
                return f"❌ Error: {str(e)}", ""
        
        def execute_full(provider, model, progress=gr.Progress()):
            try:
                results, trace_url = run_full_pipeline(provider, model, progress)
                status = "\n".join([f"✅ {stage}: {result}" for stage, result in results])
                link = f"[🔗 View Trace]({trace_url})"
                return status, link, None
            except Exception as e:
                return f"❌ Error: {str(e)}", "", None
        
        run_phase1_btn.click(
            execute_phase1,
            inputs=[pipeline_provider, pipeline_model],
            outputs=[pipeline_status, trace_link, pipeline_id_state],
        )
        
        run_phase2_btn.click(
            execute_phase2,
            inputs=[pipeline_provider, pipeline_model, pipeline_id_state],
            outputs=[pipeline_status, trace_link],
        )
        
        run_full_btn.click(
            execute_full,
            inputs=[pipeline_provider, pipeline_model],
            outputs=[pipeline_status, trace_link, pipeline_id_state],
        )


def create_data_review_tab():
    """Create the Data Review tab."""
    with gr.Tab("Data Review"):
        gr.Markdown("# 📋 Data Quality Review Report (Before Cleaning)")
        
        refresh_btn = gr.Button("🔄 Reload Report")
        report_content = gr.Markdown(load_report("data_review_report.md"))
        
        refresh_btn.click(
            lambda: load_report("data_review_report.md"),
            outputs=[report_content],
        )


def create_data_quality_tab():
    """Create the Data Quality Certificate tab (Post-Clean Report)."""
    with gr.Tab("Data Quality ✅"):
        gr.Markdown("# ✅ Data Quality Certificate (After Cleaning)")
        gr.Markdown("""
        This report is generated **after** data cleaning to verify all issues have been resolved.
        Compare this with the initial Data Review report to see the improvements.
        """)
        
        refresh_btn = gr.Button("🔄 Reload Certificate")
        report_content = gr.Markdown(load_report("data_review_post_clean.md"))
        
        refresh_btn.click(
            lambda: load_report("data_review_post_clean.md"),
            outputs=[report_content],
        )

def apply_cleaning_changes():
    """Apply changes from cleaning_proposal.csv to the database."""
    try:
        import pandas as pd
        from pymongo import MongoClient
        from bson import ObjectId
        
        filepath = REPORTS_DIR / "cleaning_proposal.csv"
        if not filepath.exists():
            return "❌ Proposal file not found. Run the Data Cleaner agent first."
        
        # Read CSV with flexible parsing to handle unquoted commas
        df = pd.read_csv(filepath, on_bad_lines='warn')
        
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        collection = db.sales
        
        updated_count = 0
        deleted_count = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                record_id = row['RecordID']
                field = str(row.get('Field', ''))
                value = str(row.get('ProposedValue', ''))
                reason = str(row.get('Reason', '')).lower()
                
                # Convert ObjectId if needed
                if isinstance(record_id, str) and len(record_id) == 24:
                    try:
                        query_id = ObjectId(record_id)
                    except:
                        query_id = record_id
                else:
                    query_id = record_id
                
                # Handle DELETE operations for duplicates
                if 'DELETE' in value.upper() or 'duplicate' in reason:
                    result = collection.delete_one({"_id": query_id})
                    if result.deleted_count > 0:
                        deleted_count += 1
                    continue
                
                # Skip rows with invalid field names
                if not field or field == 'nan' or '[' in field:
                    errors += 1
                    continue
                
                # Handle numeric conversion
                if isinstance(value, str):
                    try:
                        if '.' in value and value.replace('.', '').replace('-', '').isdigit():
                            value = float(value)
                        elif value.lstrip('-').isdigit():
                            value = int(value)
                    except:
                        pass
                
                result = collection.update_one(
                    {"_id": query_id},
                    {"$set": {field: value}}
                )
                
                if result.modified_count > 0:
                    updated_count += 1
            except Exception:
                errors += 1
                continue
        
        client.close()
        
        # Yield intermediate status before running Post-Clean Review
        yield f"✅ Updated {updated_count} records, deleted {deleted_count} duplicates. ({errors} errors)\n\n🔄 Now generating Data Quality Certificate... (this may take 1-2 minutes)"
        
        # Run Post-Clean Review to generate Data Quality Certificate
        try:
            import asyncio
            from agents import Runner
            from agents.tracing import trace, set_tracing_export_api_key
            from sales_agents.data_reviewer import create_data_reviewer
            
            if OPENAI_API_KEY:
                set_tracing_export_api_key(OPENAI_API_KEY)
            
            async def run_post_clean():
                reviewer = create_data_reviewer("deepseek", "deepseek-chat")
                with trace("Post-Clean Review") as t:
                    await Runner.run(reviewer, """Generate a comprehensive POST-CLEANING data quality report.

IMPORTANT: This is a FULL forensic audit of the CLEANED data. You must:
1. Query the `sales` collection and analyze its current state.
2. Check for any remaining issues (typos, nulls, outliers, duplicates, date problems).
3. Provide detailed statistics: record counts, null rates, outlier analysis.
4. Compare to the expected clean state and confirm all known issues were resolved.
5. Assign a Data Quality Score (0-100).

Save the report as 'data_review_post_clean.md'. This should be a detailed write-up, not just a summary.""", max_turns=40)
                return "✅ Data Quality Certificate generated!"
            
            cert_result = asyncio.run(run_post_clean())
            yield f"✅ Updated {updated_count} records, deleted {deleted_count} duplicates. ({errors} errors)\n\n{cert_result}\n\n📋 Check the 'Data Quality ✅' tab for the certificate."
        except Exception as cert_err:
            yield f"✅ Updated {updated_count} records, deleted {deleted_count} duplicates. ({errors} errors)\n\n⚠️ Could not generate certificate: {str(cert_err)}"
    except Exception as e:
        yield f"❌ Error applying changes: {str(e)}"


def create_data_cleaning_tab():
    """Create the Data Cleaning tab."""
    with gr.Tab("Data Cleaning"):
        gr.Markdown("# 🧹 Data Cleaning Log & Proposals")
        
        # Status at the top
        status_msg = gr.Textbox(label="Status", interactive=False, value="Ready - Review proposal and click Apply to proceed.")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Reload Proposal")
            apply_btn = gr.Button("✅ Apply Changes to Database", variant="primary")
        
        # Initialize download with file if it exists
        csv_path = REPORTS_DIR / "cleaning_proposal.csv"
        initial_csv = str(csv_path) if csv_path.exists() else None
        download_btn = gr.File(label="Download Proposal (CSV)", value=initial_csv)
        
        report_content = gr.Markdown(load_report("cleaning_proposal.md"))
        
        def load_proposal():
            content = load_report("cleaning_proposal.md")
            csv_path = REPORTS_DIR / "cleaning_proposal.csv"
            return content, str(csv_path) if csv_path.exists() else None, "Proposal reloaded."
        
        refresh_btn.click(
            load_proposal,
            outputs=[report_content, download_btn, status_msg],
        )
        
        apply_btn.click(
            apply_cleaning_changes,
            outputs=[status_msg],
        )


def create_analysis_tab():
    """Create the Analysis tab."""
    with gr.Tab("Analysis"):
        gr.Markdown("# 📈 Sales Analysis Report")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Reload Report")
            refresh_viz_btn = gr.Button("🔄 Reload Visualizations")
        
        report_content = gr.Markdown(load_report("analysis.md"))
        
        def get_analysis_viz():
            all_viz = get_visualizations()
            # Exclude forecast visualizations
            return [v for v in all_viz if "forecast" not in v.name.lower() and "model" not in v.name.lower()]

        gr.Markdown("### Visualizations")
        viz_gallery = gr.Gallery(
            value=get_analysis_viz(),
            label="Analysis Charts",
            columns=3,
            height="auto",
            interactive=False,
        )
        
        refresh_btn.click(
            lambda: load_report("analysis.md"),
            outputs=[report_content],
        )
        
        refresh_viz_btn.click(
            get_analysis_viz,
            outputs=[viz_gallery],
        )




def create_forecasting_tab():
    """Create the Forecasting tab."""
    with gr.Tab("Forecasting"):
        gr.Markdown("# 🔮 Sales Forecasting Report")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Reload Report")
            refresh_viz_btn = gr.Button("🔄 Reload Visualizations")
        
        report_content = gr.Markdown(load_report("forecast_report.md"))
        
        gr.Markdown("### Forecast Visualizations")
        
        # Show forecast-specific visualizations
        def get_forecast_viz():
            all_viz = get_visualizations()
            return [v for v in all_viz if "forecast" in v.name.lower() or "model" in v.name.lower() or "seasonal" in v.name.lower() or "yoy" in v.name.lower()]
        
        viz_gallery = gr.Gallery(
            value=get_forecast_viz(),
            label="Forecast Charts",
            columns=2,
            height="auto",
        )
        
        refresh_btn.click(
            lambda: load_report("forecast_report.md"),
            outputs=[report_content],
        )
        
        refresh_viz_btn.click(
            get_forecast_viz,
            outputs=[viz_gallery],
        )


def create_query_tab():
    """Create the Conversational Query tab."""
    with gr.Tab("💬 Query"):
        gr.Markdown("# 💬 Ask Questions About Your Data")
        gr.Markdown("Use natural language to explore your sales data.")
        
        with gr.Row():
            query_provider = gr.Dropdown(
                choices=list(LLM_PROVIDERS.keys()),
                value="deepseek",
                label="LLM Provider",
                scale=1,
            )
            query_model = gr.Dropdown(
                choices=get_models_for_provider("deepseek"),
                value="deepseek-chat",
                label="Model",
                scale=1,
            )
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
            type="messages",
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What were total sales in 2017?",
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        clear_btn = gr.Button("🗑️ Clear Conversation")
        trace_info = gr.Markdown("")
        
        gr.Markdown("### 💡 Example Questions")
        with gr.Row():
            ex1 = gr.Button("What were total sales?", size="sm")
            ex2 = gr.Button("Top 5 products by revenue", size="sm")
            ex3 = gr.Button("Compare sales by region", size="sm")
            ex4 = gr.Button("Which category sells best?", size="sm")
        
        # Update models when provider changes
        def update_query_models(provider):
            models = get_models_for_provider(provider)
            return gr.Dropdown(choices=models, value=models[0] if models else "")
        
        query_provider.change(update_query_models, inputs=[query_provider], outputs=[query_model])
        
        # Chat functionality
        async def respond_async(message, history, provider, model):
            from sales_agents.conversational import create_conversational_agent
            
            if not message.strip():
                return history, ""
            
            # Add user message
            history = history + [{"role": "user", "content": message}]
            
            # Get response
            if OPENAI_API_KEY:
                set_tracing_export_api_key(OPENAI_API_KEY)
            
            agent = create_conversational_agent(provider, model)
            
            # Build conversation context
            context = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" 
                for m in history[-5:]  # Last 5 messages for context
            ])
            
            conversation_id = f"chat_{uuid.uuid4().hex[:8]}"
            
            with trace(f"Query: {message[:50]}...", group_id=conversation_id) as t:
                result = await Runner.run(agent, message)
                trace_url = f"https://platform.openai.com/traces/{t.trace_id}"
            
            # Add assistant response
            history = history + [{"role": "assistant", "content": result.final_output}]
            
            trace_md = f"[🔗 View Trace]({trace_url})"
            return history, trace_md
        
        def respond(message, history, provider, model):
            return asyncio.run(respond_async(message, history, provider, model))
        
        submit_btn.click(
            respond,
            inputs=[user_input, chatbot, query_provider, query_model],
            outputs=[chatbot, trace_info],
        ).then(
            lambda: "",
            outputs=[user_input],
        )
        
        user_input.submit(
            respond,
            inputs=[user_input, chatbot, query_provider, query_model],
            outputs=[chatbot, trace_info],
        ).then(
            lambda: "",
            outputs=[user_input],
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, trace_info])
        
        # Example button handlers
        def use_example(example_text):
            return example_text
        
        ex1.click(lambda: "What were total sales?", outputs=[user_input])
        ex2.click(lambda: "Show me the top 5 products by revenue", outputs=[user_input])
        ex3.click(lambda: "Compare sales between all regions", outputs=[user_input])
        ex4.click(lambda: "Which category has the highest sales?", outputs=[user_input])


def create_settings_tab():
    """Create the Settings tab."""
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("# ⚙️ Configuration")
        
        gr.Markdown("### 🔑 API Key Status")
        
        api_status = validate_api_keys()
        status_text = "\n".join([
            f"- **{provider.capitalize()}**: {'✅ Configured' if configured else '❌ Not set'}"
            for provider, configured in api_status.items()
        ])
        gr.Markdown(status_text)
        
        gr.Markdown("""
        ### 📝 Configuration
        
        API keys are loaded from environment variables or `.env` file:
        
        ```
        OPENAI_API_KEY=sk-...
        ANTHROPIC_API_KEY=sk-ant-...
        DEEPSEEK_API_KEY=sk-...
        ```
        
        Copy `.env.example` to `.env` and add your API keys.
        """)
        
        gr.Markdown("### 🗄️ MongoDB Connection")
        gr.Textbox(value=MONGODB_URI, label="MongoDB URI", interactive=False)
        gr.Textbox(value=MONGODB_DATABASE, label="Database", interactive=False)
        
        gr.Markdown("### 📂 Output Directories")
        gr.Textbox(value=str(REPORTS_DIR), label="Reports Directory", interactive=False)
        gr.Textbox(value=str(VISUALIZATIONS_DIR), label="Visualizations Directory", interactive=False)


# =============================================================================
# Main Application
# =============================================================================

def create_app():
    """Create the Gradio application."""
    with gr.Blocks(
        title="Sales Data Analysis System",
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container {max-width: 1200px !important;}
        """,
    ) as app:
        gr.Markdown("""
        # 🛒 Multi-Agent Sales Data Analysis System
        
        Analyze ~10,000 sales records using AI-powered agents with support for 
        OpenAI, Anthropic, and DeepSeek models.
        """)
        
        create_dashboard_tab()
        create_data_review_tab()
        create_data_cleaning_tab()
        create_data_quality_tab()
        create_analysis_tab()
        create_forecasting_tab()
        create_query_tab()
        create_settings_tab()
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(OUTPUTS_DIR)],
    )
