import sys
from pathlib import Path
import random
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import MONGODB_URI, MONGODB_DATABASE

def seed_marketing_campaigns(db):
    """
    Seeds marketing campaigns that correlate with sales regions and dates.
    Uses 'target_region' and date ranges to loosely couple with sales data.
    """
    print("Seeding 'marketing_campaigns' collection...")
    collection = db.marketing_campaigns
    collection.drop()  # Reset
    
    campaigns = []
    regions = ["West", "East", "Central", "South"]
    channels = ["Social Media", "Email", "TV", "Search", "Influencer"]
    
    # Generate campaigns from 2014 to 2018
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2018, 12, 31)
    current_date = start_date
    
    campaign_id_counter = 1000
    
    while current_date < end_date:
        # Create 2-3 campaigns per month across different regions
        num_campaigns_this_month = random.randint(2, 4)
        
        for _ in range(num_campaigns_this_month):
            region = random.choice(regions)
            duration_days = random.randint(14, 45)
            camp_start = current_date + timedelta(days=random.randint(0, 20))
            camp_end = camp_start + timedelta(days=duration_days)
            
            if camp_end > end_date:
                continue
                
            budget = random.choice([5000, 10000, 25000, 50000, 75000])
            channel = random.choice(channels)
            
            campaign = {
                "campaign_id": f"CAMP-{campaign_id_counter}",
                "name": f"{channel} {region} {camp_start.strftime('%b %Y')} Push",
                "channel": channel,
                "target_region": region,
                "start_date": camp_start,
                "end_date": camp_end,
                "budget": budget,
                "conversion_target": "Sales Lift",
                "notes": f"Targeting {region} region for {channel} engagement."
            }
            campaigns.append(campaign)
            campaign_id_counter += 1
            
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    if campaigns:
        collection.insert_many(campaigns)
        print(f"Inserted {len(campaigns)} marketing campaigns.")
        # Create index for efficient joining
        collection.create_index("target_region")
        collection.create_index("start_date")
        collection.create_index("end_date")

def seed_economic_indicators(db):
    """
    Seeds monthly economic indicators (inflation, unemployment, consumer confidence).
    """
    print("Seeding 'economic_indicators' collection...")
    collection = db.economic_indicators
    collection.drop() # Reset
    
    indicators = []
    
    # Generate monthly data from 2014 to 2018
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2018, 12, 31)
    current_date = start_date
    
    # Simulating a trend
    unemployment = 6.0
    inflation = 1.5
    confidence = 90.0
    
    while current_date <= end_date:
        # Random walk for metrics
        unemployment += random.uniform(-0.1, 0.1)
        unemployment = max(3.5, min(10.0, unemployment))
        
        inflation += random.uniform(-0.05, 0.1)
        inflation = max(0.5, min(5.0, inflation))
        
        confidence += random.uniform(-2, 2)
        confidence = max(60, min(120, confidence))
        
        record = {
            "date": current_date, # First of the month
            "month_str": current_date.strftime("%Y-%m"),
            "unemployment_rate": round(unemployment, 2),
            "inflation_rate": round(inflation, 2),
            "consumer_confidence_index": round(confidence, 1),
            "gdp_growth_rate": round(random.uniform(1.5, 3.5), 1)
        }
        indicators.append(record)
        
        # Next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
            
    if indicators:
        collection.insert_many(indicators)
        print(f"Inserted {len(indicators)} economic indicator records.")
        collection.create_index("date")

def main():
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        
        seed_marketing_campaigns(db)
        seed_economic_indicators(db)
        
        print("\nContext data seeding complete!")
        print("Collections created: 'marketing_campaigns', 'economic_indicators'")
        client.close()
    except Exception as e:
        print(f"Error seeding data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
