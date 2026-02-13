
import logging
import sys
import pandas as pd
from src.influx_service import InfluxService
import src.config as config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def check_flow_rates():
    logging.info("=== Checking Flow Rate Data ===")
    
    influx = InfluxService(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG
    )
    
    # Fetch last 24 hours
    df = influx.get_training_data(lookback_hours=24)
    
    if df.empty:
        logging.error("No data found")
        return
        
    flow_col = config.FLOW_RATE_ENTITY_ID.split(".", 1)[-1]
    
    if flow_col not in df.columns:
        logging.error(f"Flow rate column {flow_col} not found in data")
        logging.info(f"Available columns: {df.columns.tolist()}")
        return
        
    flow_data = df[flow_col].dropna()
    
    if flow_data.empty:
        logging.warning("Flow rate column exists but contains no data")
        return
        
    logging.info(f"Flow Rate Stats ({flow_col}):")
    logging.info(f"  Min: {flow_data.min()}")
    logging.info(f"  Max: {flow_data.max()}")
    logging.info(f"  Mean: {flow_data.mean()}")
    logging.info(f"  Median: {flow_data.median()}")
    logging.info(f"  Count: {len(flow_data)}")
    
    # Check how many are below 100
    below_100 = flow_data[flow_data < 100]
    logging.info(f"  Count < 100: {len(below_100)} ({len(below_100)/len(flow_data)*100:.1f}%)")

if __name__ == "__main__":
    check_flow_rates()
