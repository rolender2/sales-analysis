"""
Shared function tools for all agents.

These tools use the @function_tool decorator from the OpenAI Agents SDK
to automatically generate schemas and handle validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

# Import function_tool from openai-agents SDK
from agents import function_tool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
)


# =============================================================================
# MongoDB Tools
# =============================================================================

class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB types."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _get_mongo_client() -> MongoClient:
    """Get a MongoDB client connection."""
    return MongoClient(MONGODB_URI)


@function_tool
def query_mongodb(
    collection: str,
    pipeline_json: str,
    database: str = MONGODB_DATABASE,
    limit: int = 5,
) -> dict:
    """
    Execute a MongoDB aggregation pipeline and return results.
    
    Args:
        collection: Name of the MongoDB collection to query
        pipeline_json: JSON string of aggregation pipeline stages (e.g., '[{"$match": {"Category": "Furniture"}}]')
        database: Database name (defaults to 'superstore')
        limit: Maximum number of documents to return (default 5, max 10)
        
    Returns:
        Dictionary with 'count' (total results) and 'data' (list of documents, truncated)
        
    Example pipeline_json:
        '[{"$match": {"Category": "Furniture"}}, {"$group": {"_id": "$Region", "total": {"$sum": 1}}}]'
    """
    try:
        # Parse the JSON pipeline
        pipeline = json.loads(pipeline_json)
        
        client = _get_mongo_client()
        db = client[database]
        
        # Cap limit to prevent context overflow (increased to 100 for bulk cleaning)
        limit = min(limit, 100)
        
        # Add limit stage if not present
        if not any("$limit" in stage for stage in pipeline):
            pipeline = pipeline + [{"$limit": limit}]
        
        result = list(db[collection].aggregate(pipeline))
        
        # Convert to JSON-safe format
        result_json = json.loads(json.dumps(result, cls=MongoJSONEncoder))
        
        # Truncate long string fields to save tokens
        def truncate_strings(obj, max_len=100):
            if isinstance(obj, dict):
                return {k: truncate_strings(v, max_len) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_strings(item, max_len) for item in obj]
            elif isinstance(obj, str) and len(obj) > max_len:
                return obj[:max_len] + "..."
            return obj
        
        result_json = truncate_strings(result_json)
        
        client.close()
        
        return {
            "success": True,
            "count": len(result_json),
            "data": result_json,
            "note": f"Results limited to {limit} documents. Use specific aggregations for full dataset analysis."
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON pipeline: {str(e)}",
            "count": 0,
            "data": [],
        }
    except PyMongoError as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "data": [],
        }


@function_tool
def insert_mongodb(
    collection: str,
    documents_json: str,
    database: str = MONGODB_DATABASE,
) -> dict:
    """
    Insert documents into a MongoDB collection.
    
    Args:
        collection: Name of the MongoDB collection
        documents_json: JSON string of documents to insert (e.g., '[{"field": "value"}]')
        database: Database name (defaults to 'superstore')
        
    Returns:
        Dictionary with 'success', 'inserted_count', and optionally 'error'
    """
    try:
        # Parse the JSON documents
        documents = json.loads(documents_json)
        if not isinstance(documents, list):
            documents = [documents]
        
        client = _get_mongo_client()
        db = client[database]
        
        if len(documents) == 1:
            result = db[collection].insert_one(documents[0])
            inserted_count = 1
        else:
            result = db[collection].insert_many(documents)
            inserted_count = len(result.inserted_ids)
        
        client.close()
        
        return {
            "success": True,
            "inserted_count": inserted_count,
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON documents: {str(e)}",
            "inserted_count": 0,
        }
    except PyMongoError as e:
        return {
            "success": False,
            "error": str(e),
            "inserted_count": 0,
        }


@function_tool
def get_collection_stats(
    collection: str,
    database: str = MONGODB_DATABASE,
) -> dict:
    """
    Get statistics about a MongoDB collection.
    
    Args:
        collection: Name of the MongoDB collection
        database: Database name (defaults to 'superstore')
        
    Returns:
        Dictionary with collection statistics including count, fields, and sample
    """
    try:
        client = _get_mongo_client()
        db = client[database]
        coll = db[collection]
        
        # Get document count
        count = coll.count_documents({})
        
        # Get sample document for schema inference
        sample = coll.find_one()
        fields = list(sample.keys()) if sample else []
        
        # Convert sample to JSON-safe format
        sample_json = json.loads(json.dumps(sample, cls=MongoJSONEncoder)) if sample else None
        
        client.close()
        
        return {
            "success": True,
            "collection": collection,
            "database": database,
            "document_count": count,
            "fields": fields,
            "sample_document": sample_json,
        }
    except PyMongoError as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Report Tools
# =============================================================================

@function_tool
def save_report(
    filename: str,
    content: str,
    report_type: str = "markdown",
) -> dict:
    """
    Save a report to the outputs/reports directory.
    
    Args:
        filename: Name of the file (without path, e.g., 'data_review_report.md')
        content: The report content (markdown format recommended)
        report_type: Type of report ('markdown', 'json', 'text')
        
    Returns:
        Dictionary with 'success' and 'filepath'
    """
    try:
        # Ensure proper extension
        if report_type == "markdown" and not filename.endswith(".md"):
            filename = f"{filename}.md"
        elif report_type == "json" and not filename.endswith(".json"):
            filename = f"{filename}.json"
        elif report_type == "text":
            # Allow csv to pass through without appending .txt
            if not filename.endswith(".txt") and not filename.endswith(".csv"):
                filename = f"{filename}.txt"
        
        filepath = REPORTS_DIR / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Visualization Tools
# =============================================================================

@function_tool
def save_visualization(
    chart_type: str,
    data_json: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> dict:
    """
    Generate and save a visualization.
    
    Args:
        chart_type: Type of chart ('bar', 'line', 'pie', 'hbar', 'scatter')
        data_json: JSON string with 'labels' and 'values' keys (e.g., '{"labels": ["A", "B"], "values": [1, 2]}')
        filename: Output filename (without path, e.g., 'sales_by_category.png')
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Dictionary with 'success' and 'filepath'
        
    Example data_json:
        '{"labels": ["Furniture", "Technology", "Office Supplies"], "values": [1000, 2000, 1500]}'
    """
    try:
        # Parse JSON data
        data = json.loads(data_json)
        
        # Ensure .png extension
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        
        filepath = VISUALIZATIONS_DIR / filename
        
        labels = data.get("labels", [])
        values = data.get("values", [])
        colors = data.get("colors", None)
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        
        if chart_type == "bar":
            plt.bar(labels, values, color=colors or sns.color_palette("husl", len(labels)))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")
            
        elif chart_type == "hbar":
            plt.barh(labels, values, color=colors or sns.color_palette("husl", len(labels)))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
        elif chart_type == "line":
            plt.plot(labels, values, marker="o", linewidth=2, markersize=6)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")
            
        elif chart_type == "pie":
            plt.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors or sns.color_palette("husl", len(labels)),
            )
            
        elif chart_type == "scatter":
            x_values = data.get("x", labels)
            y_values = data.get("y", values)
            plt.scatter(x_values, y_values, alpha=0.6)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        
        if title:
            plt.title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON data: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@function_tool
def create_dataframe_visualization(
    data_json: str,
    chart_type: str,
    x_column: str,
    y_column: str,
    filename: str,
    title: str = "",
    hue_column: str = "",
) -> dict:
    """
    Create a visualization from a JSON array of objects (query results).
    
    Args:
        data_json: JSON string of array of objects (e.g., '[{"category": "A", "sales": 100}]')
        chart_type: Type of chart ('bar', 'line', 'scatter', 'box')
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        filename: Output filename
        title: Chart title
        hue_column: Optional column for color grouping (empty string if not used)
        
    Returns:
        Dictionary with 'success' and 'filepath'
    """
    try:
        # Parse JSON data
        data = json.loads(data_json)
        
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        
        filepath = VISUALIZATIONS_DIR / filename
        
        df = pd.DataFrame(data)
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        
        # Handle empty hue_column
        hue = hue_column if hue_column else None
        
        if chart_type == "bar":
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue)
        elif chart_type == "line":
            sns.lineplot(data=df, x=x_column, y=y_column, hue=hue, marker="o")
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue)
        elif chart_type == "box":
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue)
        
        if title:
            plt.title(title, fontsize=14, fontweight="bold")
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON data: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@function_tool
def fix_date_formats(collection: str, target_field: str = "Order Date") -> dict:
    """
    Apply a bulk fix to standardized date formats.
    It takes strings like '15/04/2018' (Day/Month/Year) and converts them to standard '04/15/2018' (Month/Day/Year).
    
    Use this when you detect systematic "Swapped Date" issues affecting thousands of records.
    It is much more efficient than listing 5,000 updates in a CSV.
    
    Args:
        collection: The collection to update (e.g. 'sales')
        target_field: The field with date issues (e.g. 'Order Date')
        
    Returns:
        Summary of how many records were updated.
    """
    try:
        updated_count = 0
        client = _get_mongo_client()
        db = client[MONGODB_DATABASE]
        col = db[collection]
        
        # Iterate and fix
        cursor = col.find({})
        
        for doc in cursor:
            val = doc.get(target_field)
            if not isinstance(val, str):
                continue
                
            try:
                parts = val.split('/')
                if len(parts) == 3:
                    p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                    
                    # If First Part > 12, it MUST be Day. So it is DD/MM/YYYY.
                    # We want to swap to MM/DD/YYYY.
                    if p1 > 12 and p2 <= 12:
                         new_val = f"{p2:02d}/{p1:02d}/{p3}"
                         col.update_one({"_id": doc["_id"]}, {"$set": {target_field: new_val}})
                         updated_count += 1
            except:
                continue
                
        return {
            "status": "success",
            "message": f"Scanned collection. Swapped Day/Month for {updated_count} records where Day > 12.",
            "records_updated": updated_count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
