# ML Heating — Grafana Observability

Importable Grafana dashboard for full observability of the ML Heating control
system. Covers **all 8 InfluxDB measurements** in the `ml_heating_features`
bucket via the existing `homeflux` Flux datasource.

## Files

| File | Purpose |
|------|---------|
| `ml-heating-observability.json` | Importable dashboard (Grafana 12, schemaVersion 39) |

The dashboard is bound to the `homeflux` datasource through a dashboard
variable `$DS_HOMEFLUX` (uid `59kx3nmgz`). Every panel queries
`from(bucket: "ml_heating_features")` with Flux.

## Dashboard layout (10 rows, 48 panels)

1. **System Health** — one-glance stat tiles + promotion gauge
2. **OE Drift-vs-Heal** ⭐ — the new per-cycle telemetry: `outlet_effectiveness_delta`
   (drift accumulator), trajectory vs equilibrium gradients, OE + correction %,
   active/passive/stable-active cycle counts, `oe_mean_revert` state timeline
3. **Prediction Accuracy** — MAE/RMSE (1h/6h/24h), accuracy bands, improvement %
4. **Thermal Parameters** — outlet effectiveness, heat-loss coefficient, time
   constant, per-parameter stability gauges, learning rate
5. **Learning Phase** — stability score, learning weight, efficiency %, update counts
6. **Trajectory Forecast** — trajectory MAE, overshoot prevention, convergence, integration quality
7. **Shadow Benchmark** — ML outlet prediction vs heat-curve actual, energy savings, latest snapshot table
8. **Thermodynamics** — COP, thermal power, ΔT, flow rate, inlet temp
9. **Trust State** — `is_trusted` timeline, promotion progress, trust snapshot table
10. **Feature Importance** — learned physics coefficient weights (bar gauge)

## Import steps (browser)

1. Open Grafana → **Dashboards** → **New** → **Import**.
2. **Upload JSON file** → select `ml-heating-observability.json`.
3. Grafana shows the import dialog. Under **InfluxDB datasource**, confirm
   `homeflux` is selected (it should pre-fill from the dashboard variable).
4. Click **Import**. The dashboard opens at the top (System Health row).
5. Set the time range (default `now-6h`) and refresh interval (default `5m`)
   as needed.

> The dashboard UID is `ml-heating-obs`. Re-importing the same file will
> **update** the existing dashboard in place (no duplicates).

## Verification: confirm `homeflux` can read `ml_heating_features`

Before relying on the dashboard, confirm the `homeflux` datasource token has
read access to the `ml_heating_features` bucket:

1. Grafana → **Explore** → select the **homeflux** datasource.
2. Switch to **Flux** mode and run:

```flux
from(bucket: "ml_heating_features")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "ml_thermal_parameters")
  |> filter(fn: (r) => r._field == "outlet_effectiveness_delta")
  |> last()
```

3. If a row returns, the datasource has read access — the dashboard will
   populate. If you get an `unauthorized` / `access denied` error, the
   `homeflux` token needs read permission on the `ml_heating_features` bucket
   (configure in the InfluxDB datasource settings or InfluxDB token admin).

## Notes / assumptions

- The `homeflux` datasource (uid `59kx3nmgz`, Flux, org `erbehome`, endpoint
  `http://influxdb:8086`) must already exist in Grafana — it does at
  `https://grafana.erbehome.de`.
- All queries use Flux (`influxdb` datasource type). Grafana 12 supports the
  legacy InfluxDB/Flux datasource.
- No provisioning files or push scripts are included — import is manual via
  the browser, as specified.
