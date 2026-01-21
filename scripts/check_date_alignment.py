from pymongo import MongoClient
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MONGODB_URI, MONGODB_DATABASE

def check_alignment():
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    
    print("Checking Date Ranges...")
    
    # Check Sales
    sales_pipeline = [
        {"$project": {"date": {"$dateFromString": {"dateString": "$Order Date", "format": "%d/%m/%Y", "onError": None}}}},
        {"$group": {"_id": None, "min": {"$min": "$date"}, "max": {"$max": "$date"}}}
    ]
    sales_dates = list(db.sales.aggregate(sales_pipeline))
    if sales_dates and sales_dates[0]['min']:
        print(f"Sales Range: {sales_dates[0]['min']} to {sales_dates[0]['max']}")
    else:
        print("Sales Range: Could not parse dates (possibly string format issues in raw data?)")

    # Check Marketing
    marketing_dates = list(db.marketing_campaigns.aggregate([
        {"$group": {"_id": None, "min": {"$min": "$start_date"}, "max": {"$max": "$end_date"}}}
    ]))
    if marketing_dates:
        print(f"Marketing Range: {marketing_dates[0]['min']} to {marketing_dates[0]['max']}")
        
    # Check Economics
    econ_dates = list(db.economic_indicators.aggregate([
        {"$group": {"_id": None, "min": {"$min": "$date"}, "max": {"$max": "$date"}}}
    ]))
    if econ_dates:
        print(f"Economic Range: {econ_dates[0]['min']} to {econ_dates[0]['max']}")

if __name__ == "__main__":
    check_alignment()
