"""
This module provides an interface for interacting with InfluxDB.

It abstracts the complexities of writing Flux queries and handling the
InfluxDB client, offering methods to fetch historical data, PV forecasts,
and training data sets.
"""
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging
from influxdb_client import InfluxDBClient, QueryApi, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from . import config


class InfluxService:
    """A service for interacting with InfluxDB."""

    def __init__(self, url, token, org):
        """Initializes the InfluxDB client."""
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.query_api: QueryApi = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def get_pv_forecast(self) -> list[float]:
        """
        Retrieves the PV (photovoltaic) power forecast for the next 4 hours.

        It queries InfluxDB for two separate PV forecast entities, sums them,
        and then aligns the data to hourly intervals.
        """
        flux_query = """
            import "experimental"

            stop = experimental.addDuration(d: 4h, to: now())

            from(bucket: "home_assistant/autogen")
            |> range(start: -1h, stop: stop)
            |> filter(fn: (r) => r["_measurement"] == "W")
            |> filter(fn: (r) => r["_field"] == "value")
            |> filter(fn: (r) =>
                r["entity_id"] == "pvForecastWattsPV1" or
                r["entity_id"] == "pvForecastWattsPV2"
            )
            |> group()
            |> pivot(
                rowKey: ["_time"],
                columnKey: ["entity_id"],
                valueColumn: "_value"
            )
            |> map(
                fn: (r) => ({
                    _time: r._time,
                    total: (if exists r["pvForecastWattsPV1"] then
                        r["pvForecastWattsPV1"]
                    else
                        0.0) + (if exists r["pvForecastWattsPV2"] then
                        r["pvForecastWattsPV2"]
                    else
                        0.0),
                })
            )
            |> sort(columns: ["_time"])
            |> yield(name: "4h_total_forecast")
        """
        try:
            raw = self.query_api.query_data_frame(flux_query)
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

        df = (
            pd.concat(raw, ignore_index=True)
            if isinstance(raw, list)
            else raw
        )
        if df.empty or "_time" not in df.columns or "total" not in df.columns:
            return [0.0, 0.0, 0.0, 0.0]

        df["_time"] = pd.to_datetime(df["_time"], utc=True)
        df.sort_values("_time", inplace=True)

        # Align forecast to the next 4 full hours.
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
        first_anchor = now_utc.ceil("h")
        anchors = pd.date_range(start=first_anchor, periods=4, freq="h", tz="UTC")

        series = df.set_index("_time")["total"].sort_index()
        # Find the nearest forecast value for each hourly anchor.
        matched = series.reindex(
            anchors, method="nearest", tolerance=pd.Timedelta("30min")
        )
        results = [float(x) if pd.notna(x) else 0.0 for x in matched.tolist()]

        return results

    def fetch_history(
        self,
        entity_id: str,
        steps: int,
        default_value: float,
        agg_fn: str = "mean",
    ) -> list[float]:
        """
        Fetches historical data for a given entity_id.

        It retrieves data for a specified number of steps, with each step's
        duration defined in the config. The aggregation function used in the
        Flux `aggregateWindow` can be selected via `agg_fn` (e.g. "mean" or "max").
        The output is padded/resampled to a fixed length if necessary.
        """
        minutes = steps * config.HISTORY_STEP_MINUTES
        entity_id_short = entity_id.split(".", 1)[-1]

        # Sanitize aggregation function
        agg_fn = agg_fn if agg_fn in ("mean", "max", "min", "last", "first", "sum") else "mean"

        flux_query = f"""
        from(bucket: "{config.INFLUX_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["entity_id"] == "{entity_id_short}")
          |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(
            every: {config.HISTORY_STEP_MINUTES}m,
            fn: {agg_fn},
            createEmpty: false
        )
          |> pivot(
              rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value"
          )
          |> keep(columns:["_time","value"])
          |> sort(columns:["_time"])
          |> tail(n: {steps})
        """
        try:
            df = self.query_api.query_data_frame(flux_query)
            df = (
                pd.concat(df, ignore_index=True)
                if isinstance(df, list)
                else df
            )
            # Forward-fill and back-fill to handle any missing data points.
            df["value"] = df["value"].ffill().bfill()
            values = df["value"].tolist()
            
            # Ensure the result has the desired number of steps.
            if len(values) < steps:
                padding = [
                    values[-1] if values else default_value
                ] * (steps - len(values))
                values.extend(padding)
            
            # Resample the data to the exact number of steps required.
            if len(values) != steps:
                logging.debug(
                    "Resampling history from %d to %d points.",
                    len(values),
                    steps,
                )
                # Create an array of original indices
                original_indices = np.linspace(0, 1, len(values))
                # Create an array of new indices
                new_indices = np.linspace(0, 1, steps)
                # Interpolate the values at the new indices
                values = np.interp(new_indices, original_indices, values)

            return [float(v) for v in values]
        except Exception:
            # Return a default list if the query fails.
            return [default_value] * steps

    def fetch_binary_history(self, entity_id: str, steps: int) -> list[float]:
        """
        Convenience wrapper for fetching binary signals (e.g. defrost, fireplace)
        using `max` aggregation so short pulses are preserved as 1.0 in the
        aggregated windows.
        """
        return self.fetch_history(entity_id, steps, 0.0, agg_fn="max")

    def fetch_outlet_history(self, steps: int) -> list[float]:
        """Fetches the historical heating outlet temperature."""
        return self.fetch_history(
            config.ACTUAL_OUTLET_TEMP_ENTITY_ID, steps, 40.0
        )

    def fetch_indoor_history(self, steps: int) -> list[float]:
        """Fetches the historical indoor temperature."""
        return self.fetch_history(
            config.INDOOR_TEMP_ENTITY_ID, steps, 21.0
        )

    def get_training_data(self, lookback_hours: int) -> pd.DataFrame:
        """
        Fetches a comprehensive dataset for model training.

        It queries multiple entities over a specified lookback period,
        pivots the data into a wide format, and performs cleaning steps
        like filling missing values.
        """
        hp_outlet_temp_id = config.ACTUAL_OUTLET_TEMP_ENTITY_ID.split(".", 1)[
            -1
        ]
        kuche_temperatur_id = config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[
            -1
        ]
        fernseher_id = config.TV_STATUS_ENTITY_ID.split(".", 1)[-1]
        dhw_status_id = config.DHW_STATUS_ENTITY_ID.split(".", 1)[-1]
        defrost_status_id = config.DEFROST_STATUS_ENTITY_ID.split(".", 1)[-1]
        disinfection_status_id = config.DISINFECTION_STATUS_ENTITY_ID.split(
            ".", 1
        )[-1]
        dhw_boost_heater_status_id = (
            config.DHW_BOOST_HEATER_STATUS_ENTITY_ID.split(".", 1)[-1]
        )
        outdoor_temp_id = config.OUTDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        pv1_power_id = config.PV1_POWER_ENTITY_ID.split(".", 1)[-1]
        pv2_power_id = config.PV2_POWER_ENTITY_ID.split(".", 1)[-1]
        pv3_power_id = config.PV3_POWER_ENTITY_ID.split(".", 1)[-1]
        
        flux_query = f"""
            from(bucket: "{config.INFLUX_BUCKET}")
            |> range(start: -{lookback_hours}h)
            |> filter(fn: (r) => r["_field"] == "value")
            |> filter(fn: (r) =>
                r["entity_id"] == "{hp_outlet_temp_id}" or
                r["entity_id"] == "{kuche_temperatur_id}" or
                r["entity_id"] == "{outdoor_temp_id}" or
                r["entity_id"] == "{pv1_power_id}" or
                r["entity_id"] == "{pv2_power_id}" or
                r["entity_id"] == "{pv3_power_id}" or
                r["entity_id"] == "{dhw_status_id}" or
                r["entity_id"] == "{defrost_status_id}" or
                r["entity_id"] == "{disinfection_status_id}"
                or r["entity_id"] == "{dhw_boost_heater_status_id}"
                or r["entity_id"] == "{fernseher_id}"
            )
            |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
            |> pivot(
                rowKey:["_time"],
                columnKey:["entity_id"],
                valueColumn:"_value"
            )
            |> sort(columns:["_time"])
        """
        try:
            raw = self.query_api.query_data_frame(flux_query)
            df = (
                pd.concat(raw, ignore_index=True)
                if isinstance(raw, list)
                else raw
            )
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df.sort_values("_time", inplace=True)

            df.ffill(inplace=True)
            df.bfill(inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    def write_feature_importances(
        self,
        importances: dict,
        bucket: str = None,
        org: str = None,
        measurement: str = "feature_importance",
    ) -> None:
        """
        Write feature importances as a single InfluxDB point.

        The measurement will be `feature_importance` (configurable). Each
        feature name is written as a field with its importance score.

        Args:
            importances: Mapping of feature name -> importance (float).
            bucket: Target bucket name. If None, uses config.INFLUX_FEATURES_BUCKET.
            org: Influx organization. If None, uses config.INFLUX_ORG.
            measurement: Measurement name to write into.
        """
        if not importances:
            logging.debug("No importances to write to InfluxDB.")
            return

        write_bucket = (
            bucket
            or getattr(config, "INFLUX_FEATURES_BUCKET", None)
            or config.INFLUX_BUCKET
        )
        write_org = org or getattr(config, "INFLUX_ORG", None)

        try:
            p = Point(measurement).tag("source", "ml_heating")
            # Add model timestamp
            p = p.field(
                "exported", int(datetime.now(timezone.utc).timestamp())
            )
            # Add each feature as a field (field keys must be strings)
            for feature, val in importances.items():
                # Influx field names may contain dots; replace with
                # underscore for safety
                field_key = feature.replace(".", "_")
                try:
                    p = p.field(field_key, float(val))
                except Exception:
                    # If conversion fails, store 0.0 and log
                    logging.exception(
                        "Failed to convert importance for %s", feature
                    )
                    p = p.field(field_key, 0.0)

            self.write_api.write(bucket=write_bucket, org=write_org, record=p)
            logging.debug(
                "Wrote feature importances to Influx bucket '%s' "
                "(measurement=%s)",
                write_bucket,
                measurement,
            )
        except Exception as e:
            logging.exception(
                "Failed to write feature importances to InfluxDB: %s", e
            )


def create_influx_service():
    """
    Factory function to create an instance of the InfluxService.

    It reads the necessary connection details from the config module.
    """
    return InfluxService(
        config.INFLUX_URL, config.INFLUX_TOKEN, config.INFLUX_ORG
    )
