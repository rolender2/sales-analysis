# Multi-Agent Sales Data Analysis System

A comprehensive multi-agent system for analyzing sales data using the OpenAI Agents SDK. Features AI-powered data review, cleaning, analysis, forecasting, and a conversational query interface.

## 🚀 Features

- **7 Specialized Agents**:
  - **DataReviewer**: Profiles data quality and generates review reports
  - **DataCleaner**: Cleans and transforms raw data (now with bulk fix tools)
  - **Visualizer**: Dedicated agent for generating 22+ Matplotlib charts
  - **DataAnalyst (Strategist)**: Interprets charts and writes narrative reports
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
│   ├── visualizer.py         # Agent 3A (New!)
│   ├── data_analyst.py       # Agent 3B (Strategist)
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

## 🧪 Diagnostics & Chaos Testing (New!)

We have moved beyond simple descriptive analytics ("What happened?") to **Diagnostic Analytics** ("Why it happened?") and added rigorous testing tools.

### 1. Context Data Seeding
We now ingest external data to explain sales trends (`scripts/seed_context_data.py`):
- **Marketing Campaigns**: 180+ campaigns correlated with Sales Regions/Dates.
- **Economic Indicators**: Inflation, Unemployment, and Consumer Confidence data.

### 2. "Chaos Monkey" Stress Testing
To verify the Data Cleaner, we use a tool that injects intended defects (`scripts/inject_bad_data.py`):
- **Infected**: 100 random records (1% of data).
- **Defects**: Typos ("Calfornia"), Nulls, Outliers ($1M sales), Duplicates, and Bad Dates.
- **Audit**: Generates `outputs/reports/chaos_log.csv` for transparency.

---

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
- **Analyze**: Scans the `sales` collection for specific quality issues (and Chaos defects!).
- **Bulk Fix Tools**: Uses `fix_date_formats` to instantly correct thousands of date errors.
- **Propose**: Generates a CSV proposal (`cleaning_proposal.csv`) listing record-level changes.
- **Review**: Users can download the CSV from the UI for external review.
- **Apply**: Dedicated "Apply Changes" button in the UI executes the updates.

### Visualizer Agent (New)
A specialist agent dedicated purely to code generation for visualization:
- **Focused Scope**: Does not write reports or analyze text.
- **Output**: Generates **22 mandated visualizations** in `outputs/visualizations/`.
- **Chart Types**: Complex overlays (Sales vs Marketing), Heatmaps, Dual-axis plots.
- **Tools**: Matplotlib/Seaborn optimization.

### Data Analyst Agent (Strategist)
Now operates as a **Lead Strategist**, consuming the output of the Visualizer:
- **Role**: Pure diagnostic analysis and narrative writing.
- **Inputs**: Reads the visualizations generated by Agent 3A.
- **Diagnostic**: Cross-references Sales with **Marketing & Economic data** to explain the "Why".
- **Report**: Writes a **2,000-word analysis.md** with executive summary and strategic recommendations.
- **No Code**: Focusing purely on business intelligence, not plotting code.

### Forecaster Agent
Generates sophisticated sales forecasts with **Narrative Context**:
- **Multi-Model Approach**: ARIMA, Holt-Winters, Moving Average.
- **Diagnostic Narrative**: Explicitly factors in external context (e.g., "Forecast risks high due to falling consumer confidence").
- **Visualizations**: Overlays forecast with context data where possible.
- **Professional Charts**: Clean formatting, clear legends, and confidence intervals.

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
