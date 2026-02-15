import logging
import sys
import os
from datetime import datetime
from influxdb_client import InfluxDBClient

# Setup logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Add src to path
sys.path.append(os.getcwd())

from src.config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG

def check_measurements():
    target_bucket = "ml_heating_features"
    print(f"Checking InfluxDB at {INFLUX_URL}, Bucket: {target_bucket}")
    
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    
    # Query for the last 1 hour
    query = f'''
    from(bucket: "{target_bucket}")
      |> range(start: -1h)
      |> filter(fn: (r) => r["_measurement"] == "feature_importance")
      |> limit(n: 10)
    '''
    
    print("Executing query...")
    try:
        tables = query_api.query(query)
        
        record_count = 0
        for table in tables:
            for record in table.records:
                record_count += 1
                print(f"Found record: Time={record.get_time()}, Field={record.get_field()}, Value={record.get_value()}")
        
        if record_count == 0:
            print("No records found for 'feature_importance' in the last hour.")
        else:
            print(f"Total records found: {record_count}")
            
    except Exception as e:
        print(f"Query failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_measurements()
