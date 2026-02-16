"""
Standardized data loading for analysis and notebooks.
Wraps InfluxService to provide clean DataFrames for research.
"""

from typing import List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta, timezone

from .. import config
from ..influx_service import InfluxService


class DataLoader:
    """
    Centralized data access layer for analysis.
    Ensures notebooks use the same data definitions as production.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None
    ):
        """
        Initialize the data loader.
        Defaults to config values if not provided.
        """
        self.url = url or config.INFLUX_URL
        self.token = token or config.INFLUX_TOKEN
        self.org = org or config.INFLUX_ORG
        self.service = InfluxService(self.url, self.token, self.org)

    def fetch_training_data(
        self,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime] = "now()",
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch a clean DataFrame for model training/validation.

        Args:
            start_time: Start of the window (ISO string or datetime)
            end_time: End of the window (default: now)
            features: List of feature names to fetch (optional)

        Returns:
            DataFrame with datetime index and feature columns.
        """
        # Ensure start_time is datetime
        if isinstance(start_time, str):
            # Handle relative time strings if simple enough, or assume ISO
            try:
                # If it's a relative string like "-30d", this will fail,
                # but InfluxService expects datetime.
                # For now, we assume ISO format if string.
                start_time = pd.to_datetime(start_time).to_pydatetime()
            except Exception:
                # Fallback: if we can't parse it, we might need to handle
                # relative strings manually or fail. For this implementation,
                # we'll try to rely on pandas. If it fails, we might default
                # to a reasonable lookback or raise.
                print(
                    f"Warning: Could not parse start_time '{start_time}'. "
                    "Using -24h."
                )
                start_time = datetime.now(timezone.utc) - timedelta(hours=24)

        # Ensure end_time is datetime
        if isinstance(end_time, str):
            if end_time == "now()":
                end_time = datetime.now(timezone.utc)
            else:
                try:
                    end_time = pd.to_datetime(end_time).to_pydatetime()
                except Exception:
                    end_time = datetime.now(timezone.utc)
        
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        # Default features if None
        if features is None:
            features = [
                'indoor_temperature',
                'outdoor_temperature',
                'outlet_temperature',
                'ml_target_temperature'
            ]

        # Use the robust fetch_historical_data method from InfluxService
        return self.service.fetch_historical_data(
            entities=features,
            start_time=start_time,
            end_time=end_time
        )

    def close(self):
        """Close the underlying InfluxDB connection."""
        self.service.close()
