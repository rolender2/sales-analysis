import sys
from pathlib import Path
import random
import csv
from pymongo import MongoClient
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import MONGODB_URI, MONGODB_DATABASE, REPORTS_DIR

def inject_bad_data():
    """
    Injects exactly 100 defects into the sales collection and logs them.
    Types: Typos, Nulls, Outliers, Duplicates, Bad Dates.
    """
    print(" unleashing Chaos Monkey on 'sales' collection...")
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        collection = db.sales
        
        # Reset: Ensure we start with clean-ish data if possible, or just modify current
        # For this exercise, we modify existing records in place.
        
        # Get 100 random document IDs to target
        all_ids = [doc["_id"] for doc in collection.find({}, {"_id": 1})]
        if len(all_ids) < 100:
            print("Not enough records to inject 100 defects.")
            return

        target_ids = random.sample(all_ids, 150)
        
        # Define ranges
        # 0-77: Typos (77)
        # 77-107: Nulls (30)
        # 107-110: Outliers (3)
        # 110-120: Duplicates (10)
        # 120-150: Dates (30)
        
        grp_typos = target_ids[0:77]
        grp_nulls = target_ids[77:107]
        grp_outliers = target_ids[107:110]
        grp_dupes = target_ids[110:120]
        grp_dates = target_ids[120:150]
        
        chaos_log = []
        
        # 1. Typos: Variable States (77 records)
        print("Injecting 77 varied typos...")
        typo_map = {
            "California": "Californa",
            "New York": "New Yrok", 
            "Texas": "Texs",
            "Pennsylvania": "Pennsylvani",
            "Washington": "Washngton",
            "Illinois": "Illnois",
            "Ohio": "Ohhio",
            "Florida": "Florda"
        }
        for _id in grp_typos:
            doc = collection.find_one({"_id": _id})
            original = doc.get("State", "Unknown")
            
            if original in typo_map:
                bad_value = typo_map[original]
            else:
                bad_value = f"{original}_typo"
            
            collection.update_one({"_id": _id}, {"$set": {"State": bad_value}})
            chaos_log.append([str(_id), "State", original, bad_value, "typo_location"])

        # 2. Nulls: Missing Values (30 records)
        print("Injecting 30 null values...")
        for _id in grp_nulls:
            doc = collection.find_one({"_id": _id})
            field = random.choice(["Region", "Category", "Segment"])
            original = doc.get(field, "Unknown")
            
            collection.update_one({"_id": _id}, {"$set": {field: None}})
            chaos_log.append([str(_id), field, original, "None", "missing_value"])

        # 3. Outliers (3 records)
        print("Injecting 3 outliers...")
        for _id in grp_outliers:
            doc = collection.find_one({"_id": _id})
            original = doc.get("Sales", 0)
            bad_value = 999999.99
            collection.update_one({"_id": _id}, {"$set": {"Sales": bad_value}})
            chaos_log.append([str(_id), "Sales", original, bad_value, "outlier_value"])

        # 4. Duplicates (10 records)
        print("Injecting 10 duplicates...")
        for _id in grp_dupes:
            doc = collection.find_one({"_id": _id})
            del doc["_id"] 
            result = collection.insert_one(doc)
            chaos_log.append([str(result.inserted_id), "Record", "N/A", "Duplicate of " + str(_id), "duplicate_record"])

        # 5. Invalid Dates (30 records)
        print("Injecting 30 boolean dates (string format)...")
        for _id in grp_dates:
            doc = collection.find_one({"_id": _id})
            original = doc.get("Order Date", "")
            bad_value = "2025-13-01" 
            collection.update_one({"_id": _id}, {"$set": {"Order Date": bad_value}})
            chaos_log.append([str(_id), "Order Date", original, bad_value, "invalid_date"])

        # Save Log
        log_path = REPORTS_DIR / "chaos_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["RecordID", "Field", "OriginalValue", "BadValue", "DefectType"])
            writer.writerows(chaos_log)
            
        print(f"\nChaos Complete! {len(chaos_log)} defects injected.")
        print(f"Log saved to: {log_path}")
        
    except Exception as e:
        print(f"Error injecting bad data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    inject_bad_data()
