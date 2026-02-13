from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional


class SensorBuffer:
    """
    In-memory circular buffer for sensor data smoothing.

    Manages a sliding window of sensor readings to calculate rolling averages
    without querying InfluxDB during real-time inference.
    """

    def __init__(self, max_age_minutes: int = 120):
        """
        Initialize the sensor buffer.

        Args:
            max_age_minutes: Maximum age of data to keep in the buffer
                             (default: 2 hours)
        """
        self.max_age = timedelta(minutes=max_age_minutes)
        # Dictionary mapping sensor_id -> deque of (timestamp, value)
        self._buffers: Dict[str, deque[Tuple[datetime, float]]] = defaultdict(
            deque
        )

    def add_reading(
        self,
        sensor_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Add a new sensor reading to the buffer.

        Args:
            sensor_id: The ID of the sensor (e.g., 'sensor.flow_rate')
            value: The sensor value
            timestamp: The timestamp of the reading (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self._buffers[sensor_id].append((timestamp, float(value)))
        self._prune(sensor_id)

    def hydrate(self, history: Dict[str, List[Tuple[datetime, float]]]):
        """
        Bulk load historical data into the buffer (e.g., from InfluxDB).

        Args:
            history: Dictionary mapping sensor_id -> list of (timestamp, value)
        """
        for sensor_id, readings in history.items():
            # Clear existing buffer for this sensor to avoid duplicates
            self._buffers[sensor_id].clear()

            # Sort readings by timestamp just in case
            sorted_readings = sorted(readings, key=lambda x: x[0])

            for timestamp, value in sorted_readings:
                # Ensure timestamp is timezone-aware
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                self._buffers[sensor_id].append((timestamp, float(value)))

            self._prune(sensor_id)

    def get_average(
        self, sensor_id: str, window_minutes: int
    ) -> Optional[float]:
        """
        Calculate the rolling average for a sensor over the specified window.

        Args:
            sensor_id: The ID of the sensor
            window_minutes: The time window in minutes to average over

        Returns:
            The average value, or None if no data is available
        """
        if sensor_id not in self._buffers or not self._buffers[sensor_id]:
            return None

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_minutes)

        # Filter readings within the window
        # Since deque is ordered, we could optimize this, but iteration is safe
        values = [
            val for ts, val in self._buffers[sensor_id] if ts >= cutoff
        ]

        if not values:
            return None

        return sum(values) / len(values)
        
    def get_latest(self, sensor_id: str) -> Optional[float]:
        """
        Get the most recent reading for a sensor.
        
        Args:
            sensor_id: The ID of the sensor
            
        Returns:
            The latest value, or None if no data is available
        """
        if sensor_id not in self._buffers or not self._buffers[sensor_id]:
            return None
            
        return self._buffers[sensor_id][-1][1]

    def _prune(self, sensor_id: str):
        """
        Remove readings older than max_age from the buffer.
        """
        if sensor_id not in self._buffers:
            return
            
        buffer = self._buffers[sensor_id]
        if not buffer:
            return
            
        now = datetime.now(timezone.utc)
        cutoff = now - self.max_age
        
        # Remove old readings from the left
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()
