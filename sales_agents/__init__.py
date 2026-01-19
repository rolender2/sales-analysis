"""
Sales Agents package for Multi-Agent Sales Data Analysis System.

This package contains all agent implementations using the OpenAI Agents SDK:
- DataReviewer: Data profiling and quality assessment
- DataCleaner: Data cleaning and transformation
- DataAnalyst: Comprehensive sales analysis
- Forecaster: Time series forecasting
- ConversationalAgent: Natural language query interface
- Orchestrator: Pipeline coordination
"""

from sales_agents.tools import (
    query_mongodb,
    insert_mongodb,
    save_report,
    save_visualization,
    get_collection_stats,
)

__all__ = [
    "query_mongodb",
    "insert_mongodb", 
    "save_report",
    "save_visualization",
    "get_collection_stats",
]
