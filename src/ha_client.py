"""
This module provides a client for interacting with the Home Assistant API.

It abstracts the details of making HTTP requests to Home Assistant for
fetching sensor states, setting sensor values, and calling services. This
centralizes all communication with Home Assistant, making the rest of the
application cleaner and easier to manage.
"""
import logging
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# Support both package-relative and direct import for notebooks
try:
    from . import config  # Package-relative import
except ImportError:
    import config  # Direct import fallback for notebooks


class HAClient:
    """A client for interacting with the Home Assistant API."""

    def __init__(self, url: str, token: str):
        """
        Initializes the Home Assistant client.

        Args:
            url: The base URL of the Home Assistant instance
            (e.g., http://homeassistant.local:8123).
            token: A long-lived access token for authentication.
        """
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_all_states(self) -> Dict[str, Any]:
        """
        Retrieves a snapshot of all entity states from Home Assistant.

        This method is highly efficient as it fetches all states in a single
        API call, which is much faster than querying entities one by one. The
        result is cached by the caller for subsequent `get_state` calls within
        the same update cycle.

        Returns:
            A dictionary where keys are entity_ids and values are the
            corresponding state objects from Home Assistant. Returns an
            empty dictionary if the request fails.
        """
        url = f"{self.url}/api/states"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            # Create a dictionary mapping entity_id to its state object for quick lookups.
            return {entity["entity_id"]: entity for entity in resp.json()}
        except requests.RequestException as exc:
            warnings.warn(f"HA request error for all states: {exc}")
            return {}

    def get_state(
        self,
        entity_id: str,
        states_cache: Optional[Dict[str, Any]] = None,
        is_binary: bool = False,
    ) -> Optional[Any]:
        """
        Retrieves the state of a specific Home Assistant entity.

        This method prioritizes using a provided `states_cache` (from
        `get_all_states`) for efficiency. If no cache is given, it makes a
        direct API call as a fallback. It handles type conversion for
        numerical and binary sensors.

        Args:
            entity_id: The full ID of the entity (e.g., 'sensor.temperature').
            states_cache: A dictionary of all states, as returned by `get_all_states`.
            is_binary: If True, treats the state as binary ('on'/'off').

        Returns:
            The processed state of the entity (float, bool, or string),
            or None if the entity is not found or its state is invalid.
        """
        if states_cache is None:
            # Fallback to individual request if no cache is provided.
            url = f"{self.url}/api/states/{entity_id}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                warnings.warn(f"HA request error {entity_id}: {exc}")
                return None
        else:
            # Use the provided cache.
            data = states_cache.get(entity_id)

        if data is None:
            return None

        state = data.get("state")
        if state in (None, "unknown", "unavailable"):
            return None

        if is_binary:
            return state == "on"

        try:
            return float(state)
        except (TypeError, ValueError):
            return state

    def set_state(
        self,
        entity_id: str,
        value: float,
        attributes: Optional[Dict[str, Any]] = None,
        round_digits: Optional[int] = 1,
    ) -> None:
        """
        Creates or updates the state of a sensor entity in Home Assistant.

        This is the primary method for publishing the model's outputs (like
        the target temperature or performance metrics) back to Home
        Assistant, making them visible and usable in the HA frontend and
        automations.

        Args:
            entity_id: The ID of the sensor to create/update.
            value: The main state value for the sensor.
            attributes: A dictionary of additional attributes for the sensor.
            round_digits: The number of decimal places to round the state
            value to.
        """
        url = f"{self.url}/api/states/{entity_id}"

        # Round the value only if round_digits is specified.
        if round_digits is not None:
            state_value = round(value, round_digits)
        else:
            state_value = value

        # The state must be sent as a string.
        payload = {
            "state": (
                f"{state_value:.{round_digits}f}"
                if isinstance(state_value, float) and round_digits is not None
                else str(state_value)
            ),
            "attributes": attributes or {},
        }
        try:
            logging.debug("Setting HA state for %s: %s", entity_id, payload)
            requests.post(url, headers=self.headers, json=payload, timeout=10)
        except requests.RequestException as exc:
            warnings.warn(f"HA state set failed for {entity_id}: {exc}")

    def get_hourly_forecast(self) -> List[float]:
        """
        Retrieves the hourly weather forecast via the Home Assistant service.

        It calls the `weather.get_forecasts` service for the configured weather
        entity and extracts the temperature forecasts for the next few hours.

        Returns:
            A list of forecasted temperatures, or a default list of zeros if
            the call fails.
        """
        svc_url = f"{self.url}/api/services/weather/get_forecasts"
        body = {"entity_id": ["weather.openweathermap"], "type": "hourly"}

        try:
            resp = requests.post(
                svc_url,
                headers=self.headers,
                json=body,
                timeout=10,
                params={"return_response": "true"},
            )
            resp.raise_for_status()
            data = resp.json()["service_response"]
        except Exception:
            # Return a default value if the API call fails.
            return [0.0, 0.0, 0.0, 0.0]

        try:
            forecast_list = data["weather.openweathermap"]["forecast"]
        except (KeyError, TypeError):
            return [0.0, 0.0, 0.0, 0.0]

        # Extract the temperature from the first 4 forecast entries.
        result = []
        for entry in forecast_list[:4]:
            temp = entry.get("temperature") if isinstance(entry, dict) else None
            result.append(
                round(temp, 2) if isinstance(temp, (int, float)) else 0.0
            )
        return result

    def log_feature_importance(self, importances: Dict[str, float]):
        """
        Publishes the model's feature importances to a Home Assistant sensor.

        This creates a sensor (`sensor.ml_feature_importance`) where the
        state is the number of features, and an attribute `top_features`
        lists the most influential features and their importance scores. This
        is useful for monitoring and understanding the model's behavior.

        Args:
            importances: A dictionary mapping feature names to importance
            scores.
        """
        if not importances:
            return

        # Sort features by importance (descending).
        sorted_importances = sorted(
            importances.items(), key=lambda item: item[1], reverse=True
        )

        attributes = get_sensor_attributes("sensor.ml_feature_importance")
        # Store the top 10 features and their importance percentage.
        attributes["top_features"] = {
            f: round(v * 100, 2) for f, v in sorted_importances[:10]
        }
        attributes["last_updated"] = datetime.now(timezone.utc).isoformat()

        logging.debug("Logging feature importance")
        # The state of the sensor is the total number of features.
        self.set_state(
            "sensor.ml_feature_importance",
            len(sorted_importances),
            attributes,
            round_digits=None,
        )

    def log_model_metrics(self, mae: float, rmse: float) -> None:
        """
        Publishes key model performance metrics to dedicated HA sensors.

        This creates sensors for Mean Absolute Error (MAE) and Root Mean
        Squared Error (RMSE), allowing for real-time tracking of the model's
        performance from within Home Assistant.
        
        Note: Model confidence is now provided via 
        sensor.ml_heating_learning.state to avoid redundancy.

        Args:
            mae: The current Mean Absolute Error.
            rmse: The current Root Mean Squared Error.
        """
        now_utc = datetime.now(timezone.utc).isoformat()

        # Log Mean Absolute Error (MAE)
        logging.debug("Logging MAE")
        attributes_mae = get_sensor_attributes(config.MAE_ENTITY_ID)
        attributes_mae["last_updated"] = now_utc
        self.set_state(
            config.MAE_ENTITY_ID,
            mae,
            attributes_mae,
            round_digits=4,
        )

        # Log Root Mean Squared Error (RMSE)
        logging.debug("Logging RMSE")
        attributes_rmse = get_sensor_attributes(config.RMSE_ENTITY_ID)
        attributes_rmse["last_updated"] = now_utc
        self.set_state(
            config.RMSE_ENTITY_ID,
            rmse,
            attributes_rmse,
            round_digits=4,
        )

    def log_adaptive_learning_metrics(self, learning_metrics: Dict[str, Any]) -> None:
        """
        Publishes adaptive learning metrics to Home Assistant sensors.
        
        REFACTORED: Implements clean sensor schema to eliminate redundancy:
        - ml_heating_learning: Learning confidence and thermal parameters only
        - ml_model_mae: Enhanced with time-windowed attributes
        - ml_model_rmse: Enhanced with error distribution attributes
        - ml_prediction_accuracy: 24h control quality (no redundant MAE/RMSE)

        Args:
            learning_metrics: Dictionary containing all learning metrics
                from the EnhancedModelWrapper
        """
        now_utc = datetime.now(timezone.utc).isoformat()

        # 1. ML Heating Learning sensor: Confidence + thermal parameters
        attributes_learning = get_sensor_attributes("sensor.ml_heating_learning")
        attributes_learning.update({
            # Learned thermal parameters
            "thermal_time_constant": learning_metrics.get("thermal_time_constant", 6.0),
            "heat_loss_coefficient": learning_metrics.get("heat_loss_coefficient", 0.05),
            "outlet_effectiveness": learning_metrics.get("outlet_effectiveness", 0.8),
            
            # Learning progress
            "cycle_count": learning_metrics.get("cycle_count", 0),
            "parameter_updates": learning_metrics.get("parameter_updates", 0),
            "model_health": learning_metrics.get("model_health", "unknown"),
            "learning_progress": min(1.0, learning_metrics.get("cycle_count", 0) / 100.0),
            "is_improving": learning_metrics.get("is_improving", False),
            "improvement_percentage": learning_metrics.get("improvement_percentage", 0.0),
            "total_predictions": learning_metrics.get("total_predictions", 0),
            "last_updated": now_utc
        })

        # State is the learning confidence score (no redundant attribute)
        learning_confidence = learning_metrics.get("learning_confidence", 0.0)
        self.set_state(
            "sensor.ml_heating_learning",
            learning_confidence,
            attributes_learning,
            round_digits=3,
        )

        # 2. Enhanced MAE sensor: All-time MAE + time-windowed breakdowns
        attributes_mae = get_sensor_attributes("sensor.ml_model_mae")
        attributes_mae.update({
            "mae_1h": learning_metrics.get("mae_1h", 0.0),
            "mae_6h": learning_metrics.get("mae_6h", 0.0), 
            "mae_24h": learning_metrics.get("mae_24h", 0.0),
            "trend_direction": self._get_mae_trend(learning_metrics),
            "prediction_count": learning_metrics.get("total_predictions", 0),
            "last_updated": now_utc
        })

        # State is all-time MAE
        mae_all_time = learning_metrics.get("mae_all_time", 0.0)
        self.set_state(
            "sensor.ml_model_mae",
            mae_all_time,
            attributes_mae,
            round_digits=4,
        )

        # 3. Enhanced RMSE sensor: All-time RMSE + error distribution
        attributes_rmse = get_sensor_attributes("sensor.ml_model_rmse")
        attributes_rmse.update({
            "recent_max_error": learning_metrics.get("recent_max_error", 0.0),
            "std_error": self._calculate_std_error(learning_metrics),
            "mean_bias": self._calculate_mean_bias(learning_metrics),
            "prediction_count": learning_metrics.get("total_predictions", 0),
            "last_updated": now_utc
        })

        # State is all-time RMSE
        rmse_all_time = learning_metrics.get("rmse_all_time", 0.0)
        self.set_state(
            "sensor.ml_model_rmse", 
            rmse_all_time,
            attributes_rmse,
            round_digits=4,
        )

        # 4. Clean prediction accuracy sensor: 24h control quality only
        attributes_accuracy = get_sensor_attributes("sensor.ml_prediction_accuracy")
        attributes_accuracy.update({
            "perfect_accuracy_pct": learning_metrics.get("perfect_accuracy_pct", 0.0),
            "tolerable_accuracy_pct": learning_metrics.get("tolerable_accuracy_pct", 0.0),
            "poor_accuracy_pct": learning_metrics.get("poor_accuracy_pct", 0.0),
            "prediction_count_24h": learning_metrics.get("prediction_count_24h", 0),
            "excellent_all_time_pct": learning_metrics.get("excellent_accuracy_pct", 0.0),
            "good_all_time_pct": learning_metrics.get("good_accuracy_pct", 0.0),
            "last_updated": now_utc
        })

        # State is the percentage of good control (24h window, ±0.2°C)
        good_control_24h = learning_metrics.get("good_control_pct", 0.0)
        self.set_state(
            "sensor.ml_prediction_accuracy",
            good_control_24h,
            attributes_accuracy,
            round_digits=1,
        )

        logging.debug("Logged refactored sensor metrics to HA")

    def _get_mae_trend(self, metrics: Dict[str, Any]) -> str:
        """Determine MAE trend direction."""
        improvement_pct = metrics.get("improvement_percentage", 0.0)
        if improvement_pct > 5.0:
            return "improving"
        elif improvement_pct < -5.0:
            return "degrading"
        else:
            return "stable"

    def _calculate_std_error(self, metrics: Dict[str, Any]) -> float:
        """Calculate standard deviation of errors (placeholder)."""
        # This would need access to individual prediction errors
        # For now, estimate from MAE vs RMSE relationship
        mae = metrics.get("mae_all_time", 0.0)
        rmse = metrics.get("rmse_all_time", 0.0)
        if rmse > mae:
            return round((rmse**2 - mae**2)**0.5, 4)
        return 0.0

    def _calculate_mean_bias(self, metrics: Dict[str, Any]) -> float:
        """Calculate mean bias (systematic over/under-prediction)."""
        # This would need access to individual prediction errors with sign
        # For now, return 0.0 as placeholder
        return 0.0


def get_sensor_attributes(entity_id: str) -> Dict[str, Any]:
    """
    Provides a standardized set of attributes for the sensors created by this script.

    This function acts as a central repository for sensor metadata like
    `friendly_name`, `unit_of_measurement`, `device_class`, and `icon`.
    This ensures that all sensors created by the application have a
    consistent and user-friendly appearance in the Home Assistant frontend.

    Args:
        entity_id: The ID of the sensor for which to get attributes.

    Returns:
        A dictionary of attributes for that sensor.
    """
    base_attributes = {
        "state_class": "measurement",
    }
    sensor_specific_attributes = {
        "sensor.ml_vorlauftemperatur": {
            "unique_id": "ml_heating_target_outlet_temp",
            "friendly_name": "ML Target Outlet Temp",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "icon": "mdi:thermometer-water",
        },
        "sensor.ml_predicted_indoor_temp": {
            "unique_id": "ml_heating_predicted_indoor_temp",
            "friendly_name": "ML Predicted Indoor Temp",
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "icon": "mdi:home-thermometer-outline",
        },
        "sensor.ml_model_confidence": {
            "unique_id": "ml_heating_model_confidence",
            "friendly_name": "ML Model Confidence",
            "unit_of_measurement": "std dev",
            "icon": "mdi:chart-line",
        },
        "sensor.ml_model_mae": {
            "unique_id": "ml_heating_model_mae",
            "friendly_name": "ML Model MAE",
            "unit_of_measurement": "°C",
            "icon": "mdi:chart-line",
        },
        "sensor.ml_model_rmse": {
            "unique_id": "ml_heating_model_rmse",
            "friendly_name": "ML Model RMSE",
            "unit_of_measurement": "°C",
            "icon": "mdi:chart-line",
        },
        "sensor.ml_feature_importance": {
            "unique_id": "ml_heating_feature_importance",
            "friendly_name": "ML Feature Importance",
            "unit_of_measurement": "%",
            "icon": "mdi:format-list-numbered",
        },
        "sensor.ml_heating_state": {
            "unique_id": "ml_heating_state",
            "friendly_name": "ML Heating State",
            "unit_of_measurement": "state",
            "icon": "mdi:information-outline",
        },
        "sensor.ml_heating_learning": {
            "unique_id": "ml_heating_learning",
            "friendly_name": "ML Adaptive Learning Metrics",
            "unit_of_measurement": "metrics",
            "icon": "mdi:brain",
        },
        "sensor.ml_prediction_accuracy": {
            "unique_id": "ml_prediction_accuracy", 
            "friendly_name": "ML Prediction Accuracy",
            "unit_of_measurement": "%",
            "icon": "mdi:target",
        },
    }
    attributes = base_attributes.copy()
    attributes.update(sensor_specific_attributes.get(entity_id, {}))
    return attributes


def create_ha_client():
    """
    Factory function to create an instance of the HAClient.

    It simplifies the creation of a client by reading the required URL and
    token directly from the application's configuration module.
    """
    return HAClient(config.HASS_URL, config.HASS_TOKEN)
