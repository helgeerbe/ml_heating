import pytest
from datetime import datetime, timedelta, timezone
from src.sensor_buffer import SensorBuffer

class TestSensorBuffer:
    def test_initialization(self):
        buffer = SensorBuffer(max_age_minutes=60)
        assert buffer.max_age == timedelta(minutes=60)
        assert len(buffer._buffers) == 0

    def test_add_reading(self):
        buffer = SensorBuffer()
        now = datetime.now(timezone.utc)
        
        buffer.add_reading("sensor.test", 10.0, now)
        
        assert len(buffer._buffers["sensor.test"]) == 1
        assert buffer._buffers["sensor.test"][0] == (now, 10.0)

    def test_add_reading_default_timestamp(self):
        buffer = SensorBuffer()
        buffer.add_reading("sensor.test", 10.0)
        
        assert len(buffer._buffers["sensor.test"]) == 1
        # Check that timestamp is recent (within last second)
        ts, val = buffer._buffers["sensor.test"][0]
        assert val == 10.0
        assert (datetime.now(timezone.utc) - ts).total_seconds() < 1.0

    def test_pruning_old_data(self):
        buffer = SensorBuffer(max_age_minutes=10)
        now = datetime.now(timezone.utc)
        
        # Add old reading (15 mins ago)
        old_ts = now - timedelta(minutes=15)
        buffer.add_reading("sensor.test", 5.0, old_ts)
        
        # Add new reading (now)
        buffer.add_reading("sensor.test", 10.0, now)
        
        # Should only have the new reading
        assert len(buffer._buffers["sensor.test"]) == 1
        assert buffer._buffers["sensor.test"][0][1] == 10.0

    def test_get_average(self):
        buffer = SensorBuffer()
        now = datetime.now(timezone.utc)
        
        # Add readings: 10, 20, 30
        buffer.add_reading("sensor.test", 10.0, now - timedelta(minutes=5))
        buffer.add_reading("sensor.test", 20.0, now - timedelta(minutes=3))
        buffer.add_reading("sensor.test", 30.0, now - timedelta(minutes=1))
        
        # Average over last 10 mins should be (10+20+30)/3 = 20
        avg = buffer.get_average("sensor.test", window_minutes=10)
        assert avg == 20.0
        
        # Average over last 2 mins should be 30 (only last reading)
        avg_short = buffer.get_average("sensor.test", window_minutes=2)
        assert avg_short == 30.0

    def test_get_average_no_data(self):
        buffer = SensorBuffer()
        assert buffer.get_average("sensor.test", 10) is None

    def test_get_latest(self):
        buffer = SensorBuffer()
        now = datetime.now(timezone.utc)
        
        buffer.add_reading("sensor.test", 10.0, now - timedelta(minutes=5))
        buffer.add_reading("sensor.test", 20.0, now)
        
        assert buffer.get_latest("sensor.test") == 20.0
        assert buffer.get_latest("sensor.unknown") is None

    def test_hydrate(self):
        buffer = SensorBuffer()
        now = datetime.now(timezone.utc)
        
        history = {
            "sensor.test": [
                (now - timedelta(minutes=10), 10.0),
                (now - timedelta(minutes=5), 20.0)
            ]
        }
        
        buffer.hydrate(history)
        
        assert len(buffer._buffers["sensor.test"]) == 2
        assert buffer.get_latest("sensor.test") == 20.0
        assert buffer.get_average("sensor.test", 15) == 15.0

    def test_hydrate_clears_existing(self):
        buffer = SensorBuffer()
        now = datetime.now(timezone.utc)
        
        buffer.add_reading("sensor.test", 99.0, now)
        
        history = {
            "sensor.test": [
                (now - timedelta(minutes=10), 10.0)
            ]
        }
        
        buffer.hydrate(history)
        
        assert len(buffer._buffers["sensor.test"]) == 1
        assert buffer.get_latest("sensor.test") == 10.0
