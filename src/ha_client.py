"""This module contains the Home Assistant client."""
import logging
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from . import config


class HAClient:
    """A client for interacting with the Home Assistant API."""

    def __init__(self, url: str, token: str):
        """Initialize the client."""
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_all_states(self) -> Dict[str, Any]:
        """Fetches all states from Home Assistant in a single API call."""
        url = f"{self.url}/api/states"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            # Create a dictionary mapping entity_id to its state object
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
        Retrieves the state of a Home Assistant entity, using a cache if provided.
        This version matches the logic of the original ml_heating.py script.
        """
        if states_cache is None:
            # Fallback to individual request if no cache is provided
            url = f"{self.url}/api/states/{entity_id}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                warnings.warn(f"HA request error {entity_id}: {exc}")
                return None
        else:
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
        """Posts a state to the Home Assistant API, with optional rounding."""
        url = f"{self.url}/api/states/{entity_id}"

        # Round the value only if round_digits is specified
        if round_digits is not None:
            state_value = round(value, round_digits)
        else:
            state_value = value

        # Ensure the state is a string with a dot for the decimal separator
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
        """Get hourly weather forecast."""
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
            return [0.0, 0.0, 0.0, 0.0]

        try:
            forecast_list = data["weather.openweathermap"]["forecast"]
        except (KeyError, TypeError):
            return [0.0, 0.0, 0.0, 0.0]

        result = []
        for entry in forecast_list[:4]:
            temp = entry.get("temperature") if isinstance(entry, dict) else None
            result.append(
                round(temp, 2) if isinstance(temp, (int, float)) else 0.0
            )
        return result

    def log_feature_importance(self, importances: Dict[str, float]):
        """Logs feature importances to a Home Assistant sensor."""
        if not importances:
            return

        sorted_importances = sorted(
            importances.items(), key=lambda item: item[1], reverse=True
        )

        attributes = get_sensor_attributes("sensor.ml_feature_importance")
        attributes["top_features"] = {
            f: round(v * 100, 2) for f, v in sorted_importances[:10]
        }
        attributes["last_updated"] = datetime.now(timezone.utc).isoformat()

        logging.debug("Logging feature importance")
        self.set_state(
            "sensor.ml_feature_importance",
            len(sorted_importances),
            attributes,
            round_digits=None,
        )

    def log_model_metrics(
        self, confidence: float, mae: float, rmse: float
    ) -> None:
        """Logs model metrics to Home Assistant sensors."""
        now_utc = datetime.now(timezone.utc).isoformat()

        # Log Confidence
        logging.debug("Logging confidence")
        attributes_confidence = get_sensor_attributes(
            config.CONFIDENCE_ENTITY_ID
        )
        attributes_confidence["last_updated"] = now_utc
        self.set_state(
            config.CONFIDENCE_ENTITY_ID,
            confidence,
            attributes_confidence,
            round_digits=4,
        )

        # Log MAE
        logging.debug("Logging MAE")
        attributes_mae = get_sensor_attributes(config.MAE_ENTITY_ID)
        attributes_mae["last_updated"] = now_utc
        self.set_state(
            config.MAE_ENTITY_ID,
            mae,
            attributes_mae,
            round_digits=4,
        )

        # Log RMSE
        logging.debug("Logging RMSE")
        attributes_rmse = get_sensor_attributes(config.RMSE_ENTITY_ID)
        attributes_rmse["last_updated"] = now_utc
        self.set_state(
            config.RMSE_ENTITY_ID,
            rmse,
            attributes_rmse,
            round_digits=4,
        )


def get_sensor_attributes(entity_id: str) -> Dict[str, Any]:
    """Returns a dictionary of attributes for a given sensor entity_id."""
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
    }
    attributes = base_attributes.copy()
    attributes.update(sensor_specific_attributes.get(entity_id, {}))
    return attributes


def create_ha_client():
    """Create a Home Assistant client."""
    return HAClient(config.HASS_URL, config.HASS_TOKEN)
