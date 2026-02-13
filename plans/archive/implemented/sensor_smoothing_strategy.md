# In-Memory Sensor Smoothing Strategy

## 1. Objective
Decouple real-time inference from InfluxDB by implementing an in-memory circular buffer for sensor data. This ensures that `InfluxService` is used *only* for historical batch retrieval (training/calibration) and `HAClient` is used *exclusively* for real-time data, while still enabling data smoothing (rolling averages) for volatile sensors.

## 2. Architecture Overview

### 2.1. The `SensorBuffer` Class
A new class `SensorBuffer` will be created (likely in `src/sensor_buffer.py` or `src/sensor_data_manager.py`) to manage a sliding window of sensor readings.

**Key Responsibilities:**
*   **Storage**: Maintain a fixed-size deque (or list) of timestamped readings for critical sensors (Flow Rate, Power, Temperatures).
*   **Ingestion**: Accept new readings from `HAClient` during the control loop.
*   **Hydration**: Accept a batch of historical data (from InfluxDB) *only* during system startup to pre-fill the buffer.
*   **Calculation**: Provide methods to calculate rolling averages (e.g., `get_rolling_average(sensor_id, window_minutes)`).

**Data Structure:**
```python
class SensorBuffer:
    def __init__(self, max_age_minutes: int = 120):
        # Dictionary mapping sensor_id -> deque of (timestamp, value)
        self._buffers: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(deque)
        self.max_age = timedelta(minutes=max_age_minutes)

    def add_reading(self, sensor_id: str, value: float, timestamp: datetime):
        # Add new reading
        # Prune old readings > max_age
        pass

    def hydrate(self, history: Dict[str, List[Tuple[datetime, float]]]):
        # Bulk load history (clears existing buffer first?)
        pass

    def get_average(self, sensor_id: str, window_minutes: int) -> float:
        # Calculate average of values within the last window_minutes
        pass
```

### 2.2. Startup Hydration (The "Cold Start" Fix)
To prevent the system from needing 1-2 hours to build up a valid rolling average after a restart:
1.  **On Startup**: `main.py` initializes `InfluxService`.
2.  **Query**: Fetch the last 2 hours of data for critical sensors (Flow Rate, Power, Inlet/Outlet Temp).
3.  **Hydrate**: Pass this data to `SensorBuffer.hydrate()`.
4.  **Disconnect**: `InfluxService` is not used again for the main control loop.

### 2.3. Fallback Mechanism (Cold Start Mode)
If InfluxDB is unavailable or the hydration query fails:
1.  **Log Warning**: Record the failure but do *not* crash the application.
2.  **Cold Start Mode**: The `SensorBuffer` starts empty.
3.  **Progressive Filling**: As the main loop runs, `HAClient` pushes new readings into the buffer.
4.  **Adaptive Smoothing**:
    *   If the buffer has fewer samples than the requested window (e.g., 5 mins of data for a 15-min window), calculate the average of *available* data.
    *   This means the first few cycles will use "noisier" data (shorter average), but the system remains operational.
    *   After `window_minutes` have passed, the smoothing becomes fully effective.

### 2.4. Real-time Control Loop Integration
In `main.py`:
1.  **Fetch**: `HAClient` gets current states.
2.  **Update**: Push current values into `SensorBuffer`.
3.  **Feature Build**: Pass the `SensorBuffer` instance to `build_physics_features`.
4.  **Predict**: Model uses smoothed features from the buffer.

### 2.5. Refactoring `physics_features.py`
*   **Remove**: Direct dependency on `InfluxService` for real-time features.
*   **Add**: Dependency on `SensorBuffer`.
*   **Logic Change**: Instead of using raw `flow_rate` or `power`, call `sensor_buffer.get_average('flow_rate', window=15)`.

## 3. Implementation Steps

### Step 1: Create `SensorBuffer` Class
*   Implement `src/sensor_buffer.py`.
*   Unit tests for adding readings, pruning old data, and calculating averages.

### Step 2: Implement Hydration Logic
*   Add a method to `InfluxService` to fetch the specific "recent history" batch needed for hydration.
*   Update `main.py` to perform this fetch once at startup.

### Step 3: Integrate into `main.py` Loop
*   Instantiate `SensorBuffer` in `main()`.
*   Update the loop to feed `HAClient` data into the buffer.
*   Implement the try-except block around hydration to handle InfluxDB failures gracefully.

### Step 4: Refactor `physics_features.py`
*   Update `build_physics_features` signature to accept `SensorBuffer`.
*   Replace instantaneous readings with smoothed values for:
    *   `flow_rate` (e.g., 10-min avg)
    *   `power_consumption` (e.g., 10-min avg)
    *   `delta_t` (derived from smoothed inlet/outlet)

### Step 5: Verify & Test
*   Verify that `InfluxService` is NOT called during the loop.
*   Verify that features are stable (smoothed) compared to raw HA readings.
*   Verify that the system starts up correctly even if InfluxDB is unreachable (simulated failure).

## 4. Specific Smoothing Requirements
*   **Flow Rate**: High frequency noise (PWM modulation). Needs ~10-15 min rolling average.
*   **Power**: Spiky. Needs ~10-15 min rolling average.
*   **Temperatures**: Slow moving, but `delta_t` is sensitive to noise. 5-10 min smoothing recommended.

## 5. Constraints & Edge Cases
*   **Missing Data**: If buffer is empty (Influx down at startup), fallback to instantaneous value.
*   **Stale Data**: If HA stops updating, buffer might contain old data. Ensure timestamps are checked against `now()`.
