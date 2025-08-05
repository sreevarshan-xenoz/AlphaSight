#!/usr/bin/env python3
"""
Data Collection Infrastructure Demo

This script demonstrates the complete data collection infrastructure
including NSE data collection, news data collection, and storage system.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nifty_ml_pipeline.data import (
    NSEDataCollector,
    NewsDataCollector,
    DataStorage,
    DataCache
)


def main():
    """Demonstrate the data collection infrastructure."""
    print("=== NIFTY 50 ML Pipeline - Data Collection Demo ===\n")
    
    # Initialize components
    print("1. Initializing data collection components...")
    nse_collector = NSEDataCollector(max_retries=2, base_delay=0.5)
    news_collector = NewsDataCollector(max_retries=2, base_delay=0.5)
    storage = DataStorage(base_path="demo_data", rolling_window_days=365)
    cache = DataCache(max_size=100)
    
    print("   ✓ NSE Data Collector initialized")
    print("   ✓ News Data Collector initialized")
    print("   ✓ Data Storage initialized")
    print("   ✓ Data Cache initialized\n")
    
    # Define date range for data collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 7 days
    
    print(f"2. Collecting data from {start_date.date()} to {end_date.date()}...")
    
    # Collect NSE data
    print("   Collecting NSE price data...")
    try:
        # Note: This will use mock data in the current implementation
        # In production, this would fetch real data from NSE
        price_df = nse_collector.collect_data('NIFTY 50', start_date, end_date)
        price_data_objects = nse_collector.convert_to_price_data_objects(price_df, 'NIFTY 50')
        print(f"   ✓ Collected {len(price_data_objects)} price records")
    except Exception as e:
        print(f"   ✗ Failed to collect NSE data: {str(e)}")
        price_data_objects = []
    
    # Collect news data
    print("   Collecting financial news data...")
    try:
        news_df = news_collector.collect_data('NIFTY 50', start_date, end_date)
        news_data_objects = news_collector.convert_to_news_data_objects(news_df)
        print(f"   ✓ Collected {len(news_data_objects)} news articles")
    except Exception as e:
        print(f"   ✗ Failed to collect news data: {str(e)}")
        news_data_objects = []
    
    print()
    
    # Store data
    print("3. Storing data with partitioning...")
    if price_data_objects:
        success = storage.store_price_data(price_data_objects, 'NIFTY 50')
        if success:
            print("   ✓ Price data stored successfully")
        else:
            print("   ✗ Failed to store price data")
    
    if news_data_objects:
        success = storage.store_news_data(news_data_objects)
        if success:
            print("   ✓ News data stored successfully")
        else:
            print("   ✗ Failed to store news data")
    
    print()
    
    # Retrieve data
    print("4. Retrieving stored data...")
    retrieved_price_data = storage.retrieve_price_data('NIFTY 50', start_date, end_date)
    retrieved_news_data = storage.retrieve_news_data(start_date, end_date)
    
    print(f"   ✓ Retrieved {len(retrieved_price_data)} price records")
    print(f"   ✓ Retrieved {len(retrieved_news_data)} news records")
    
    # Display sample data
    if retrieved_price_data:
        print("\n   Sample price data:")
        sample_price = retrieved_price_data[0]
        print(f"     Symbol: {sample_price.symbol}")
        print(f"     Date: {sample_price.timestamp.date()}")
        print(f"     Open: {sample_price.open}, Close: {sample_price.close}")
        print(f"     Volume: {sample_price.volume:,}")
    
    if retrieved_news_data:
        print("\n   Sample news data:")
        sample_news = retrieved_news_data[0]
        print(f"     Source: {sample_news.source}")
        print(f"     Date: {sample_news.timestamp.date()}")
        print(f"     Headline: {sample_news.headline[:80]}...")
    
    print()
    
    # Demonstrate caching
    print("5. Demonstrating data caching...")
    cache_key = f"price_data_NIFTY50_{start_date.date()}"
    cache.put(cache_key, retrieved_price_data)
    
    cached_data = cache.get(cache_key)
    if cached_data:
        print(f"   ✓ Cached and retrieved {len(cached_data)} price records")
    else:
        print("   ✗ Failed to cache/retrieve data")
    
    print(f"   Cache size: {cache.size()} items")
    print()
    
    # Storage statistics
    print("6. Storage statistics...")
    stats = storage.get_storage_stats()
    print(f"   Price data files: {stats['price_data_files']}")
    print(f"   News data files: {stats['news_data_files']}")
    print(f"   Total storage size: {stats['total_size_mb']:.2f} MB")
    print()
    
    # Cleanup demonstration
    print("7. Demonstrating data cleanup...")
    print("   (This would remove data outside the rolling window)")
    cleanup_success = storage.cleanup_old_data()
    if cleanup_success:
        print("   ✓ Cleanup completed successfully")
    else:
        print("   ✗ Cleanup failed")
    
    print("\n=== Demo completed successfully! ===")
    print("\nNote: This demo uses mock data for demonstration purposes.")
    print("In production, real NSE and news data would be collected.")


if __name__ == "__main__":
    main()