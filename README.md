# Multi-Agent Sales Data Analysis System

A comprehensive multi-agent system for analyzing sales data using the OpenAI Agents SDK. Features AI-powered data review, cleaning, analysis, forecasting, and a conversational query interface.

## 🚀 Features

- **6 Specialized Agents**:
  - **DataReviewer**: Profiles data quality and generates review reports
  - **DataCleaner**: Cleans and transforms raw data
  - **DataAnalyst**: Performs comprehensive sales analysis with visualizations
  - **Forecaster**: Generates forecasts using ARIMA, Holt-Winters, and Moving Average
  - **ConversationalAgent**: Natural language query interface
  - **Orchestrator**: Coordinates the multi-agent pipeline

- **Multi-LLM Support**: OpenAI, Anthropic (Claude), and DeepSeek via LiteLLM
- **Built-in Tracing**: Full observability via OpenAI Traces Dashboard
- **Interactive Frontend**: Gradio-based UI with 7 tabs
- **MongoDB Integration**: Direct database connectivity for ~10,000 sales records

## 📋 Prerequisites

- Python 3.10+
- MongoDB running locally at `mongodb://localhost:27017`
- Database: `superstore` with `sales` collection
- API keys for at least one LLM provider

## 🛠️ Installation

1. **Clone/navigate to the project**:
   ```bash
   cd /home/robert/Coding/sales
   ```

2. **Activate conda environment**:
   ```bash
   conda activate ai
   ```

3. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## 🔧 Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
MONGODB_URI=mongodb://localhost:27017
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4-turbo
OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false
```

## 🚀 Usage

### check Setup

Verify your configuration:

```bash
python main.py --check-setup
```

### Run Full Pipeline

Execute all agents in sequence (Data Review → Cleaning → Analysis → Forecasting). 
**Note:** Pipeline now runs each agent with an isolated context to prevent token overflow, while linking them via a shared Trace Group ID.

```bash
python main.py --run-pipeline
```

With DeepSeek (Cost-Effective & Optimized):

```bash
python main.py --run-pipeline --provider deepseek --model deepseek-chat
```

### Run Individual Agent

```bash
python main.py --run-agent data_reviewer
python main.py --run-agent data_analyst --provider openai --model gpt-4o
```
### Launch Gradio UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

## 📊 Gradio Interface

The web interface includes 7 tabs:

| Tab | Description |
|-----|-------------|
| **Dashboard** | KPIs, stats, and pipeline runner |
| **Data Review** | Data quality report viewer |
| **Data Cleaning** | Interactive proposal workflow with **Download CSV** and **Apply Changes** buttons |
| **Analysis** | Sales analysis report with inline visualizations (supports split reports automatically) |
| **Forecasting** | Forecast report and model comparisons |
| **Query** | Conversational AI interface for data questions |
| **Settings** | Configuration and status information |

## 🔍 Tracing

All agent executions are automatically traced. View traces at:

**https://platform.openai.com/traces**

The system uses **Isolated Contexts with Grouped Traces**:
- Each agent gets a fresh context window (preventing token overflow)
- All agents in a pipeline run share a `group_id` for unified viewing
- Supports OpenAI, Anthropic, and DeepSeek via LiteLLM

## 📁 Project Structure

```
sales/
├── sales_agents/
│   ├── __init__.py
│   ├── tools.py              # Shared function tools (optimized schemas)
│   ├── data_reviewer.py      # Agent 1
│   ├── data_cleaner.py       # Agent 2
│   ├── data_analyst.py       # Agent 3
│   ├── forecaster.py         # Agent 4
│   ├── conversational.py     # Agent 5
│   └── orchestrator.py       # Agent 6
├── outputs/
│   ├── reports/              # Generated markdown reports
│   └── visualizations/       # Generated PNG charts
├── config.py                 # Configuration module
├── main.py                   # CLI entry point
├── app.py                    # Gradio frontend
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 📈 Generated Outputs

After running the pipeline:

**Reports** (`outputs/reports/`):
- `data_review_report.md` - Data quality assessment
- `cleaning_proposal.csv` - Record-level cleaning proposals (downloadable)
- `cleaning_proposal.md` - Cleaning proposal summary
- `analysis.md` or `analysis_part*.md` - Comprehensive sales analysis (auto-merged if split)
- `forecast_report.md` - Forecasting methodology and results
- `executive_summary.md` - High-level summary (when using orchestrator)

**Visualizations** (`outputs/visualizations/`):
- `sales_by_category.png` - Category breakdown
- `sales_by_region.png` - Regional distribution
- `sales_trend_monthly.png` - Monthly trends
- `top_products.png` - Top performing products
- `forecast_overall.png` - Forecast with confidence intervals
- `model_comparison.png` - Forecast model metrics

**MongoDB Collections**:
- `sales` - Original raw data (9,800 records), updated in-place by Data Cleaner
- `analysis_results` - Analysis metrics
- `sales_forecasts` - Forecast predictions
- `query_history` - Conversation logs

## 🤖 Agent Details

### Data Reviewer Agent
Profiles the raw `sales` collection and generates a quality report including:
- Record counts and schema validation
- Missing value analysis
- Duplicate detection
- Date range coverage
- Sales statistics

### Data Cleaner Agent
Refactored to support an **Interactive Proposal Workflow**:
- **Analyze**: Scans the `sales` collection for quality issues.
- **Propose**: Generates a CSV proposal (`cleaning_proposal.csv`) listing record-level changes (e.g., string-to-number conversions).
- **Review**: Users can download the CSV from the UI for external review (Excel).
- **Apply**: Dedicated "Apply Changes" button in the UI executes the approved updates on the live database.

### Data Analyst Agent
Performs comprehensive sales analysis with a rigorous "Senior Data Analyst" persona:
- **Exploratory Analysis**: Batch processing of profitability, segmentation, and distribution metrics.
- **Deep Visualization**: Generates **18+ distinct visualizations**, optimized for readability (compact 300px width, 6x4 dimensions).
- **Robust Commentary**: Provides detailed, 3-4 paragraph insights per observation, highlighting trends, outliers, and business implications.
- **Business Reporting**: Produces a structured `analysis.md` with Executive Summary, Methodology, Detailed Findings, and Strategic Recommendations.

### Forecaster Agent
Generates sophisticated sales forecasts and educational reports:
- **Multi-Model Approach**:
  - **ARIMA (1,1,1)**: Baseline for trend detection.
  - **Holt-Winters Additive**: Captures seasonality.
  - **Moving Average (window=3)**: Simple smoothing baseline.
- **Multiple Visualizations**: The agent decides what charts are most insightful (forecast, model comparison, decomposition, etc.).
- **Educational Reporting**: Explains *why* models were chosen and interprets confidence intervals.
- **Professional Charts**: All visualizations include clear titles, axis labels, legends, and units.

### Conversational Agent
Answers natural language questions by querying the **raw `sales` collection** (~9,800 records):
- "What were total sales in 2017?"
- "Top 5 products by revenue"
- "Compare sales between regions"

## 🔒 Security Notes

- API keys should be stored in `.env` (not committed to git)
- The `.env` file is ignored by default
- Sensitive data can be excluded from traces via `trace_include_sensitive_data=False`

## 📝 License

MIT License
