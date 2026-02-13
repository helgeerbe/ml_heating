# Active Sampling Strategy for Sub-Cycle Sensor Smoothing

## Problem Statement
The current heating controller operates on a 30-minute cycle (`CYCLE_INTERVAL_MINUTES`). Between cycles, the application sleeps (or polls for blocking events like DHW/Defrost) but does not actively ingest sensor data into its internal `SensorBuffer`.

This creates a "blind spot" of up to 30 minutes. When the next cycle starts:
1. The `SensorBuffer` only contains data from *previous* cycles (30 mins ago, 60 mins ago, etc.).
2. The single instantaneous reading at the start of the new cycle is added.
3. Any high-frequency dynamics (e.g., rapid flow rate changes, short cycling, sudden temperature drops) that occurred *during* the 30-minute wait are missing from the buffer.

While InfluxDB might capture this data (if configured independently), the application's internal decision-making relies on the `SensorBuffer` for smoothing and feature engineering. Relying solely on InfluxDB queries at the start of every cycle adds latency and dependency.

## Solution: Active Sampling
We will modify the idle loop (`poll_for_blocking` in `HeatingController`) to actively sample critical sensors and populate the `SensorBuffer` while waiting for the next cycle.

### Mechanism
1.  **Inject SensorBuffer**: Pass the `SensorBuffer` instance into `poll_for_blocking`.
2.  **Sample During Poll**: The `poll_for_blocking` method already calls `ha_client.get_all_states()` every `BLOCKING_POLL_INTERVAL_SECONDS` (default 60s) to check for DHW/Defrost.
3.  **Extract & Buffer**: Inside this loop, we will extract values for critical sensors (Indoor Temp, Outlet Temp, Flow Rate, etc.) from the `all_states` dictionary and push them into the `SensorBuffer`.

### Benefits
*   **High-Resolution History**: The buffer will now contain minute-by-minute data (or whatever the polling interval is) instead of 30-minute snapshots.
*   **Better Smoothing**: Rolling averages (e.g., "avg flow rate over last 15 mins") will be accurate and representative of the inter-cycle behavior.
*   **Transient Detection**: We can detect if the system was unstable *between* cycles.

### Implementation Details

#### 1. `src/heating_controller.py`
Update `poll_for_blocking` signature:
```python
def poll_for_blocking(
    self,
    ha_client: HAClient,
    state: SystemState,
    sensor_buffer: Optional[SensorBuffer] = None  # New argument
) -> None:
```

Inside the loop:
```python
if sensor_buffer:
    # Extract values and add to buffer
    # ...
```

#### 2. `src/main.py`
Update the call site:
```python
blocking_manager.poll_for_blocking(ha_client, state, sensor_buffer)
```

### Sensors to Sample
*   `sensor.modbus_indoor_temperature`
*   `sensor.modbus_lwt` (Actual Outlet)
*   `sensor.modbus_target_temperature` (Target Outlet)
*   `sensor.outdoor_temperature`
*   `sensor.modbus_ewt` (Inlet)
*   `sensor.modbus_flow_rate`
