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

# Support both package-relative and direct import for notebooks
try:
    from . import config  # Package-relative import
except ImportError:
    import config  # Direct import fallback for notebooks


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

    def fetch_historical_data(
        self,
        entities: list[str],
        start_time: datetime,
        end_time: datetime,
        freq: str = "30min"
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple entities over a time range.
        
        This method supports the notebook pattern where multiple entities
        are fetched over a specific datetime range.
        
        Args:
            entities: List of entity names (can include domain prefix or generic names)
            start_time: Start datetime
            end_time: End datetime  
            freq: Resampling frequency (default "30min")
            
        Returns:
            DataFrame with time index and entity columns
        """
        # Calculate lookback hours from time range
        time_delta = end_time - start_time
        lookback_hours = int(time_delta.total_seconds() / 3600)
        
        # Map generic entity names to actual config entity IDs
        entity_mapping = {
            'indoor_temperature': config.INDOOR_TEMP_ENTITY_ID,
            'outdoor_temperature': config.OUTDOOR_TEMP_ENTITY_ID,
            'outlet_temperature': config.ACTUAL_OUTLET_TEMP_ENTITY_ID,
            'pv_power': config.PV_POWER_ENTITY_ID,
            'dhw_heating': config.DHW_STATUS_ENTITY_ID,
            'heat_pump_heating': config.ACTUAL_OUTLET_TEMP_ENTITY_ID,  # Same as outlet
            'ml_target_temperature': 'sensor.ml_target_temperature',  # Typical ML target
        }
        
        # Map entities to real entity IDs, fallback to original if not mapped
        real_entities = []
        for entity in entities:
            mapped_entity = entity_mapping.get(entity, entity)
            real_entities.append(mapped_entity)
        
        # Strip domain prefixes if present
        entity_ids_short = []
        original_names = []  # Keep track for column mapping later
        for i, entity in enumerate(real_entities):
            if "." in entity:
                entity_ids_short.append(entity.split(".", 1)[-1])
            else:
                entity_ids_short.append(entity)
            original_names.append(entities[i])  # Keep original name for mapping
        
        # Debug: log entity processing for troubleshooting
        logging.debug(f"Original entities: {entities}")
        logging.debug(f"Mapped to real entities: {real_entities}")
        logging.debug(f"Stripped entity IDs: {entity_ids_short}")
        
        # BULLETPROOF APPROACH: Query each entity separately to avoid OR operator issues
        # This completely sidesteps the Flux syntax problems by using proven single-entity queries
        logging.debug("Using separate queries approach to avoid OR operator issues")
        
        # Query each entity separately using proven single-entity method
        entity_dataframes = []
        for i, entity_short in enumerate(entity_ids_short):
            original_entity = entities[i]
            real_entity = real_entities[i]
            
            try:
                # Use single entity query (we know this works)
                flux_query = f"""
                from(bucket: "{config.INFLUX_BUCKET}")
                |> range(start: -{lookback_hours}h)
                |> filter(fn: (r) => r["_field"] == "value")
                |> filter(fn: (r) => r["entity_id"] == "{entity_short}")
                |> aggregateWindow(every: {freq}, fn: mean, createEmpty: false)
                |> pivot(
                    rowKey: ["_time"],
                    columnKey: ["entity_id"],
                    valueColumn: "_value"
                )
                |> sort(columns: ["_time"])
                """
                
                logging.debug(f"Querying entity: {entity_short}")
                raw = self.query_api.query_data_frame(flux_query)
                df_single = (
                    pd.concat(raw, ignore_index=True)
                    if isinstance(raw, list)
                    else raw
                )
                
                if not df_single.empty:
                    df_single["_time"] = pd.to_datetime(df_single["_time"], utc=True)
                    
                    # Rename column to expected name
                    entity_lower = original_entity.lower()
                    if "indoor" in entity_lower:
                        new_name = "indoor_temperature"
                    elif "outdoor" in entity_lower:
                        new_name = "outdoor_temperature"
                    elif ("outlet" in entity_lower or "flow" in entity_lower):
                        new_name = "outlet_temperature"
                    elif ("pv" in entity_lower or "power" in entity_lower):
                        new_name = "pv_power"
                    elif ("dhw" in entity_lower and "heat" in entity_lower):
                        new_name = "dhw_heating"
                    elif ("heat" in entity_lower and "pump" in entity_lower):
                        new_name = "heat_pump_heating"
                    elif "target" in entity_lower:
                        new_name = "ml_target_temperature"
                    elif "mode" in entity_lower:
                        new_name = "ml_control_mode"
                    else:
                        new_name = original_entity.replace(".", "_")
                    
                    # Rename the data column
                    if entity_short in df_single.columns:
                        df_single.rename(columns={entity_short: new_name}, inplace=True)
                    
                    entity_dataframes.append(df_single)
                    
            except Exception as e:
                logging.warning(f"Failed to query entity {entity_short}: {e}")
                continue
        
        # Combine all entity dataframes
        if not entity_dataframes:
            return pd.DataFrame()
        
        # Start with the first dataframe
        result_df = entity_dataframes[0].copy()
        
        # Merge additional dataframes on time
        for df_additional in entity_dataframes[1:]:
            result_df = pd.merge(result_df, df_additional, on="_time", how="outer")
        
        # Sort by time and clean up
        result_df.sort_values("_time", inplace=True)
        result_df.set_index("_time", inplace=True)
        result_df.index.name = "time"
        result_df.reset_index(inplace=True)
        
        # Fill missing values
        result_df.ffill(inplace=True)
        result_df.bfill(inplace=True)
        
        logging.debug(f"Successfully combined {len(entity_dataframes)} entities into DataFrame with shape {result_df.shape}")
        return result_df

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
        pv_power_id = config.PV_POWER_ENTITY_ID.split(".", 1)[-1]
        
        flux_query = f"""
            from(bucket: "{config.INFLUX_BUCKET}")
            |> range(start: -{lookback_hours}h)
            |> filter(fn: (r) => r["_field"] == "value")
            |> filter(fn: (r) =>
                r["entity_id"] == "{hp_outlet_temp_id}" or
                r["entity_id"] == "{kuche_temperatur_id}" or
                r["entity_id"] == "{outdoor_temp_id}" or
                r["entity_id"] == "{pv_power_id}" or
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
        timestamp: datetime = None,
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
            timestamp: Optional datetime object to use as the point's timestamp.
                       Defaults to the current UTC time if not provided.
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
            # Use provided timestamp or current UTC time
            point_time = timestamp if timestamp else datetime.now(timezone.utc)
            p = Point(measurement).tag("source", "ml_heating").time(point_time)
            
            # Add model exported field as string representation of timestamp if we use an actual timestamp for the point.
            p = p.field("exported", point_time.isoformat())

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
                "(measurement=%s) with timestamp %s",
                write_bucket, measurement, point_time.isoformat()
            )
        except Exception as e:
            logging.exception(
                "Failed to write feature importances to InfluxDB: %s", e
            )

    def write_prediction_metrics(
        self,
        prediction_metrics: dict,
        bucket: str = None,
        org: str = None,
        timestamp: datetime = None,
    ) -> None:
        """
        Write prediction accuracy metrics to InfluxDB.
        
        Exports MAE/RMSE tracking data and prediction accuracy percentages
        for the adaptive learning system monitoring.
        
        Args:
            prediction_metrics: Dict from PredictionMetrics.get_metrics()
            bucket: Target bucket name. If None, uses config.INFLUX_BUCKET.
            org: Influx organization. If None, uses config.INFLUX_ORG.
            timestamp: Optional datetime for the point timestamp.
        """
        if not prediction_metrics:
            logging.debug("No prediction metrics to write to InfluxDB.")
            return

        write_bucket = bucket or config.INFLUX_BUCKET
        write_org = org or getattr(config, "INFLUX_ORG", None)

        try:
            point_time = timestamp if timestamp else datetime.now(timezone.utc)
            p = Point("ml_prediction_metrics").tag("source", "ml_heating").tag("version", "2.0").time(point_time)
            
            # MAE metrics for different time windows
            if "1h" in prediction_metrics:
                p = p.field("mae_1h", float(prediction_metrics["1h"].get("mae", 0.0)))
                p = p.field("rmse_1h", float(prediction_metrics["1h"].get("rmse", 0.0)))
                
            if "6h" in prediction_metrics:
                p = p.field("mae_6h", float(prediction_metrics["6h"].get("mae", 0.0)))
                p = p.field("rmse_6h", float(prediction_metrics["6h"].get("rmse", 0.0)))
                
            if "24h" in prediction_metrics:
                p = p.field("mae_24h", float(prediction_metrics["24h"].get("mae", 0.0)))
                p = p.field("rmse_24h", float(prediction_metrics["24h"].get("rmse", 0.0)))
                
            if "all" in prediction_metrics:
                p = p.field("total_predictions", int(prediction_metrics["all"].get("count", 0)))

            # Prediction accuracy breakdown
            accuracy_breakdown = prediction_metrics.get("accuracy_breakdown", {})
            if accuracy_breakdown:
                p = p.field("accuracy_excellent_pct", float(accuracy_breakdown.get("excellent", {}).get("percentage", 0.0)))
                p = p.field("accuracy_very_good_pct", float(accuracy_breakdown.get("very_good", {}).get("percentage", 0.0)))
                p = p.field("accuracy_good_pct", float(accuracy_breakdown.get("good", {}).get("percentage", 0.0)))
                p = p.field("accuracy_acceptable_pct", float(accuracy_breakdown.get("acceptable", {}).get("percentage", 0.0)))

            # Trend analysis
            trends = prediction_metrics.get("trends", {})
            if trends and not trends.get("insufficient_data"):
                p = p.field("mae_improvement_pct", float(trends.get("mae_improvement_percentage", 0.0)))
                p = p.field("is_improving", bool(trends.get("is_improving", False)))

            # Calculate 24h prediction count (approximate)
            total_predictions = prediction_metrics.get("all", {}).get("count", 0)
            predictions_24h = min(288, total_predictions)  # Max 288 predictions in 24h (5min intervals)
            p = p.field("predictions_24h", int(predictions_24h))

            self.write_api.write(bucket=write_bucket, org=write_org, record=p)
            
            logging.debug(
                "Wrote prediction metrics to InfluxDB bucket '%s' with MAE(1h)=%.3f",
                write_bucket, 
                prediction_metrics.get("1h", {}).get("mae", 0.0)
            )
            
        except Exception as e:
            logging.exception("Failed to write prediction metrics to InfluxDB: %s", e)

    def write_thermal_learning_metrics(
        self,
        thermal_model,
        bucket: str = None,
        org: str = None,
        timestamp: datetime = None,
    ) -> None:
        """
        Write thermal model learning metrics to InfluxDB.
        
        Exports current thermal parameters, learning progress, and
        adaptive learning effectiveness metrics.
        
        Args:
            thermal_model: ThermalEquilibriumModel instance with learning data
            bucket: Target bucket name. If None, uses config.INFLUX_BUCKET.
            org: Influx organization. If None, uses config.INFLUX_ORG.
            timestamp: Optional datetime for the point timestamp.
        """
        try:
            # Get adaptive learning metrics from thermal model
            learning_metrics = thermal_model.get_adaptive_learning_metrics()
            
            if learning_metrics.get('insufficient_data'):
                logging.debug("Insufficient thermal learning data for export.")
                return

            write_bucket = bucket or config.INFLUX_BUCKET
            write_org = org or getattr(config, "INFLUX_ORG", None)
            point_time = timestamp if timestamp else datetime.now(timezone.utc)

            # Write thermal parameters measurement
            p = Point("ml_thermal_parameters").tag("source", "ml_heating").tag("parameter_type", "current").time(point_time)
            
            # Core thermal parameters
            current_params = learning_metrics.get('current_parameters', {})
            p = p.field("outlet_effectiveness", float(current_params.get('outlet_effectiveness', thermal_model.outlet_effectiveness)))
            p = p.field("heat_loss_coefficient", float(current_params.get('heat_loss_coefficient', thermal_model.heat_loss_coefficient)))
            p = p.field("thermal_time_constant", float(current_params.get('thermal_time_constant', thermal_model.thermal_time_constant)))
            
            # Learning metadata
            p = p.field("learning_confidence", float(learning_metrics.get('learning_confidence', 1.0)))
            p = p.field("current_learning_rate", float(learning_metrics.get('current_learning_rate', 0.01)))
            p = p.field("parameter_updates_total", int(learning_metrics.get('parameter_updates', 0)))
            
            # Calculate parameter corrections as percentages (if baseline data available)
            try:
                baseline_effectiveness = getattr(thermal_model, 'base_outlet_effectiveness', thermal_model.outlet_effectiveness)
                correction_pct = ((thermal_model.outlet_effectiveness - baseline_effectiveness) / baseline_effectiveness) * 100
                p = p.field("outlet_effectiveness_correction_pct", float(correction_pct))
            except (AttributeError, ZeroDivisionError):
                p = p.field("outlet_effectiveness_correction_pct", 0.0)

            # Parameter stability metrics
            p = p.field("thermal_time_constant_stability", float(learning_metrics.get('thermal_time_constant_stability', 0.0)))
            p = p.field("heat_loss_coefficient_stability", float(learning_metrics.get('heat_loss_coefficient_stability', 0.0)))
            p = p.field("outlet_effectiveness_stability", float(learning_metrics.get('outlet_effectiveness_stability', 0.0)))
            
            # Calculate 24h parameter updates
            parameter_updates_24h = min(learning_metrics.get('parameter_updates', 0), 288)  # Max 288 updates in 24h
            p = p.field("parameter_updates_24h", int(parameter_updates_24h))

            self.write_api.write(bucket=write_bucket, org=write_org, record=p)
            
            logging.debug(
                "Wrote thermal learning metrics to InfluxDB: effectiveness=%.3f, confidence=%.2f",
                thermal_model.outlet_effectiveness,
                learning_metrics.get('learning_confidence', 1.0)
            )
            
        except Exception as e:
            logging.exception("Failed to write thermal learning metrics to InfluxDB: %s", e)

    def write_learning_phase_metrics(
        self,
        learning_phase_data: dict,
        bucket: str = None,
        org: str = None,
        timestamp: datetime = None,
    ) -> None:
        """
        Write learning phase classification metrics to InfluxDB.
        
        Exports current learning phase, stability scores, and learning
        distribution statistics for hybrid learning monitoring.
        
        Args:
            learning_phase_data: Dict with learning phase information
            bucket: Target bucket name. If None, uses config.INFLUX_BUCKET.
            org: Influx organization. If None, uses config.INFLUX_ORG.
            timestamp: Optional datetime for the point timestamp.
        """
        if not learning_phase_data:
            logging.debug("No learning phase data to write to InfluxDB.")
            return

        write_bucket = bucket or config.INFLUX_BUCKET
        write_org = org or getattr(config, "INFLUX_ORG", None)

        try:
            point_time = timestamp if timestamp else datetime.now(timezone.utc)
            current_phase = learning_phase_data.get('current_learning_phase', 'unknown')
            
            p = Point("ml_learning_phase").tag("source", "ml_heating").tag("learning_phase", current_phase).time(point_time)
            
            # Current learning state
            p = p.field("current_learning_phase", str(current_phase))
            p = p.field("stability_score", float(learning_phase_data.get('stability_score', 0.0)))
            p = p.field("learning_weight_applied", float(learning_phase_data.get('learning_weight_applied', 0.0)))
            p = p.field("stable_period_duration_min", int(learning_phase_data.get('stable_period_duration_min', 0)))
            
            # Learning distribution (24h counts)
            learning_updates_24h = learning_phase_data.get('learning_updates_24h', {})
            p = p.field("high_confidence_updates_24h", int(learning_updates_24h.get('high_confidence', 0)))
            p = p.field("low_confidence_updates_24h", int(learning_updates_24h.get('low_confidence', 0)))
            p = p.field("skipped_updates_24h", int(learning_updates_24h.get('skipped', 0)))
            
            # Learning effectiveness metrics
            p = p.field("learning_efficiency_pct", float(learning_phase_data.get('learning_efficiency_pct', 0.0)))
            p = p.field("correction_stability", float(learning_phase_data.get('correction_stability', 0.0)))
            p = p.field("false_learning_prevention_pct", float(learning_phase_data.get('false_learning_prevention_pct', 0.0)))

            self.write_api.write(bucket=write_bucket, org=write_org, record=p)
            
            logging.debug(
                "Wrote learning phase metrics to InfluxDB: phase=%s, stability=%.2f",
                current_phase,
                learning_phase_data.get('stability_score', 0.0)
            )
            
        except Exception as e:
            logging.exception("Failed to write learning phase metrics to InfluxDB: %s", e)

    def write_trajectory_prediction_metrics(
        self,
        trajectory_data: dict,
        bucket: str = None,
        org: str = None,
        timestamp: datetime = None,
    ) -> None:
        """
        Write trajectory prediction metrics to InfluxDB.
        
        Exports trajectory accuracy, overshoot prevention, and forecast
        integration quality metrics for monitoring trajectory predictions.
        
        Args:
            trajectory_data: Dict with trajectory prediction metrics
            bucket: Target bucket name. If None, uses config.INFLUX_BUCKET.
            org: Influx organization. If None, uses config.INFLUX_ORG.
            timestamp: Optional datetime for the point timestamp.
        """
        if not trajectory_data:
            logging.debug("No trajectory prediction data to write to InfluxDB.")
            return

        write_bucket = bucket or config.INFLUX_BUCKET
        write_org = org or getattr(config, "INFLUX_ORG", None)

        try:
            point_time = timestamp if timestamp else datetime.now(timezone.utc)
            prediction_horizon = trajectory_data.get('prediction_horizon', '4h')
            
            p = Point("ml_trajectory_prediction").tag("source", "ml_heating").tag("prediction_horizon", prediction_horizon).time(point_time)
            
            # Trajectory accuracy by different horizons
            trajectory_accuracy = trajectory_data.get('trajectory_accuracy', {})
            p = p.field("trajectory_mae_1h", float(trajectory_accuracy.get('mae_1h', 0.0)))
            p = p.field("trajectory_mae_2h", float(trajectory_accuracy.get('mae_2h', 0.0)))
            p = p.field("trajectory_mae_4h", float(trajectory_accuracy.get('mae_4h', 0.0)))
            
            # Overshoot prevention metrics
            overshoot_data = trajectory_data.get('overshoot_prevention', {})
            p = p.field("overshoot_predicted", bool(overshoot_data.get('overshoot_predicted', False)))
            p = p.field("overshoot_prevented_24h", int(overshoot_data.get('prevented_24h', 0)))
            p = p.field("undershoot_prevented_24h", int(overshoot_data.get('undershoot_prevented_24h', 0)))
            
            # Convergence analysis
            convergence_data = trajectory_data.get('convergence', {})
            p = p.field("convergence_time_avg_min", float(convergence_data.get('avg_time_minutes', 0.0)))
            p = p.field("convergence_accuracy_pct", float(convergence_data.get('accuracy_percentage', 0.0)))
            
            # Forecast integration quality
            forecast_data = trajectory_data.get('forecast_integration', {})
            p = p.field("weather_forecast_available", bool(forecast_data.get('weather_available', False)))
            p = p.field("pv_forecast_available", bool(forecast_data.get('pv_available', False)))
            p = p.field("forecast_integration_quality", float(forecast_data.get('quality_score', 0.0)))

            self.write_api.write(bucket=write_bucket, org=write_org, record=p)
            
            logging.debug(
                "Wrote trajectory prediction metrics to InfluxDB: horizon=%s, mae_1h=%.3f",
                prediction_horizon,
                trajectory_accuracy.get('mae_1h', 0.0)
            )
            
        except Exception as e:
            logging.exception("Failed to write trajectory prediction metrics to InfluxDB: %s", e)


def create_influx_service():
    """
    Factory function to create an instance of the InfluxService.

    It reads the necessary connection details from the config module.
    """
    return InfluxService(
        config.INFLUX_URL, config.INFLUX_TOKEN, config.INFLUX_ORG
    )
