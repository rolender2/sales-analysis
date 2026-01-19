"""
Configuration module for Multi-Agent Sales Data Analysis System.

This module provides centralized configuration for:
- MongoDB connection settings
- LLM provider configurations (OpenAI, Anthropic, DeepSeek)
- Output directory paths
- Tracing settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Ensure output directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MongoDB Configuration
# =============================================================================

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = "superstore"
MONGODB_COLLECTIONS = {
    "sales": "sales",                      # Original data
    "sales_cleaned": "sales_cleaned",      # Cleaned data
    "analysis_results": "analysis_results", # Analysis outputs
    "sales_forecasts": "sales_forecasts",  # Forecast data
    "forecast_performance": "forecast_performance",  # Model metrics
    "query_history": "query_history",      # Conversation logs
}

# =============================================================================
# LLM Provider Configuration
# =============================================================================

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Default provider settings
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo")

# Provider-specific model configurations
LLM_PROVIDERS = {
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "anthropic": {
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        "default_model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 4096,
        "temperature": 0.7,
        # LiteLLM model prefix for OpenAI Agents SDK
        "litellm_prefix": "litellm/anthropic/",
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-coder"],
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 4096,
        "temperature": 0.7,
        # LiteLLM model prefix for OpenAI Agents SDK
        "litellm_prefix": "litellm/deepseek/",
    },
}

# =============================================================================
# Tracing Configuration
# =============================================================================

# Enable tracing (sends to OpenAI Traces Dashboard)
TRACING_ENABLED = True

# Include sensitive data in traces (inputs/outputs)
TRACE_INCLUDE_SENSITIVE_DATA = os.getenv(
    "OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", "false"
).lower() in ("true", "1")

# Workflow name for traces
TRACE_WORKFLOW_NAME = "Sales Analysis Pipeline"

# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIG = {
    "data_reviewer": {
        "name": "DataReviewer",
        "description": "Profiles and assesses data quality",
    },
    "data_cleaner": {
        "name": "DataCleaner", 
        "description": "Cleans and transforms raw data",
    },
    "data_analyst": {
        "name": "DataAnalyst",
        "description": "Performs comprehensive sales analysis",
    },
    "forecaster": {
        "name": "Forecaster",
        "description": "Generates sales forecasts using multiple models",
    },
    "conversational": {
        "name": "ConversationalAgent",
        "description": "Answers natural language queries about the data",
    },
    "orchestrator": {
        "name": "Orchestrator",
        "description": "Coordinates the multi-agent pipeline",
    },
}

# =============================================================================
# Forecasting Configuration
# =============================================================================

FORECAST_CONFIG = {
    "horizon_months": 12,  # Default forecast horizon
    "train_test_split": 0.8,  # 80% train, 20% test
    "confidence_level": 0.95,  # 95% confidence intervals
    "models": ["arima", "holt_winters", "moving_average"],
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_string(provider: str, model: str) -> str:
    """
    Get the full model string for the OpenAI Agents SDK.
    
    For OpenAI models, returns the model name directly.
    For other providers, prepends the LiteLLM prefix.
    
    Args:
        provider: The LLM provider name (openai, anthropic, deepseek)
        model: The model name
        
    Returns:
        Full model string for use with Agent()
    """
    if provider == "openai":
        return model
    
    provider_config = LLM_PROVIDERS.get(provider, {})
    prefix = provider_config.get("litellm_prefix", f"litellm/{provider}/")
    return f"{prefix}{model}"


def get_available_models(provider: str) -> list[str]:
    """Get list of available models for a provider."""
    return LLM_PROVIDERS.get(provider, {}).get("models", [])


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY),
    }
