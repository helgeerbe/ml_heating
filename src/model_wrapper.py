"""This module contains the model wrapper and related functions."""
import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from river import compose, drift, metrics, preprocessing
from river.forest import ARFRegressor

from . import config
from .feature_builder import build_features_for_training, get_feature_names


def create_model() -> compose.Pipeline:
    """Create a new machine learning model."""
    feature_names = get_feature_names()

    unscaled_features = [
        "outlet_temp",
        "outlet_temp_sq",
        "outlet_temp_cub",
        "outlet_temp_change_from_last",
        "outlet_indoor_diff",
        "outdoor_temp_x_outlet_temp",
    ]

    scaled_features = [f for f in feature_names if f not in unscaled_features]

    # Create a pipeline for scaling a subset of features
    scaler = compose.Select(*scaled_features) | preprocessing.StandardScaler()

    # Create a pipeline that just passes the unscaled features through
    passthrough = compose.Select(*unscaled_features)

    # Combine them using TransformerUnion (the '+' operator is a shorthand)
    preprocessor = scaler + passthrough

    model_pipeline = compose.Pipeline(
        ("features", preprocessor),
        (
            "learn",
            ARFRegressor(
                n_models=10,
                seed=42,
                drift_detector=drift.PageHinkley(),
                warning_detector=drift.ADWIN(),
            ),
        ),
    )
    return model_pipeline


def load_model() -> Tuple[compose.Pipeline, metrics.MAE, metrics.RMSE]:
    """Load the model and metrics from a file, or create new ones."""
    try:
        with open(config.MODEL_FILE, "rb") as f:
            saved_data = pickle.load(f)
            if isinstance(saved_data, dict):
                model = saved_data["model"]
                mae = saved_data.get("mae", metrics.MAE())
                rmse = saved_data.get("rmse", metrics.RMSE())
                logging.info(
                    "Successfully loaded model and metrics from %s",
                    config.MODEL_FILE,
                )
            else:
                model = saved_data
                mae = metrics.MAE()
                rmse = metrics.RMSE()
                logging.info(
                    "Successfully loaded old format model from %s",
                    config.MODEL_FILE,
                )
            return model, mae, rmse
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        logging.warning(
            "Could not load model from %s, creating a new one.",
            config.MODEL_FILE,
        )
        return create_model(), metrics.MAE(), metrics.RMSE()


def save_model(
    model: compose.Pipeline, mae: metrics.MAE, rmse: metrics.RMSE
) -> None:
    """Save the model and metrics to a file."""
    try:
        with open(config.MODEL_FILE, "wb") as f:
            pickle.dump({"model": model, "mae": mae, "rmse": rmse}, f)
            logging.debug(
                "Successfully saved model and metrics to %s", config.MODEL_FILE
            )
    except Exception as e:
        logging.error(
            "Failed to save model and metrics to %s: %s",
            config.MODEL_FILE,
            e,
        )


def initial_train(
    model: compose.Pipeline,
    mae: metrics.MAE,
    rmse: metrics.RMSE,
    influx_service,
) -> None:
    """Train the model on historical data."""
    logging.info("--- Initial Model Training ---")
    df = influx_service.get_training_data(lookback_hours=168)
    if df.empty or len(df) < 240:  # Need sufficient data
        logging.warning("Not enough data for initial training. Skipping.")
        return

    features_list, labels_list = [], []

    for idx in range(12, len(df) - config.PREDICTION_HORIZON_STEPS):
        features = build_features_for_training(df, idx)
        if features is None:
            continue

        current_indoor = df.iloc[idx].get(
            config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        )
        future_indoor = df.iloc[idx + config.PREDICTION_HORIZON_STEPS].get(
            config.INDOOR_TEMP_ENTITY_ID.split(".", 1)[-1]
        )

        if pd.isna(current_indoor) or pd.isna(future_indoor):
            continue

        actual_delta = float(future_indoor) - float(current_indoor)
        features_list.append(features)
        labels_list.append(actual_delta)

    if features_list:
        logging.info("Training with %d samples", len(features_list))

        # We let the scaler and the regressor learn online. We give the model a
        # head start before tracking metrics, as the very first predictions
        # can be unstable.
        logging.info("Performing initial model training...")
        regressor_warm_up_samples = 100
        for i, (features, label) in enumerate(zip(features_list, labels_list)):
            y_pred = model.predict_one(features)
            model.learn_one(features, label)

            if i >= regressor_warm_up_samples:
                mae.update(label, y_pred)
                rmse.update(label, y_pred)

        logging.info("Initial training MAE: %.4f", mae.get())
        logging.info("Initial training RMSE: %.4f", rmse.get())
    else:
        logging.warning("No valid training samples found.")


def find_best_outlet_temp(
    model: compose.Pipeline,
    features: pd.DataFrame,
    current_temp: float,
    target_temp: float,
    baseline_outlet_temp: float,
    prediction_history: list,
    outlet_history: list[float],
) -> tuple[float, float, list]:
    """Find the best outlet temperature to reach the target indoor temperature."""
    logging.debug("--- Finding Best Outlet Temp ---")
    logging.debug(f"Target indoor temp: {target_temp:.1f}°C")
    best_temp, min_diff = baseline_outlet_temp, float("inf")

    x_base = features.to_dict(orient="records")[0]

    # --- Confidence Monitoring ---
    regressor = model.steps["learn"]
    tree_preds = [tree.predict_one(x_base) for tree in regressor]
    confidence = np.std(tree_preds)

    if confidence > config.CONFIDENCE_THRESHOLD:
        logging.warning(
            "Model confidence low (%.3f > %.3f), falling back to baseline.",
            confidence,
            config.CONFIDENCE_THRESHOLD,
        )
        return baseline_outlet_temp, confidence, prediction_history

    search_radius = 20.0
    min_search_temp = max(18.0, baseline_outlet_temp - search_radius)
    max_search_temp = min(55.0, baseline_outlet_temp + search_radius)
    step = 0.5

    if min_search_temp > max_search_temp:
        return baseline_outlet_temp, 0.0, prediction_history

    last_outlet_temp = outlet_history[-1]

    for temp_candidate in np.arange(
        min_search_temp, max_search_temp + step, step
    ):
        x_candidate = x_base.copy()
        x_candidate["outlet_temp"] = temp_candidate
        x_candidate["outlet_temp_sq"] = temp_candidate**2
        x_candidate["outlet_temp_cub"] = temp_candidate**3
        x_candidate["outlet_temp_change_from_last"] = (
            temp_candidate - last_outlet_temp
        )
        x_candidate["outlet_indoor_diff"] = temp_candidate - current_temp
        x_candidate["outdoor_temp_x_outlet_temp"] = (
            x_base["outdoor_temp"] * temp_candidate
        )

        predicted_delta = model.predict_one(x_candidate)
        predicted_indoor = current_temp + predicted_delta

        logging.debug(
            f"  - Test {temp_candidate:.1f}°C -> "
            f"Pred ΔT: {predicted_delta:.3f}°C, "
            f"Indoor: {predicted_indoor:.2f}°C"
        )
        diff = abs(predicted_indoor - target_temp)
        if diff < min_diff:
            min_diff, best_temp = diff, temp_candidate
        elif diff == min_diff:
            if abs(
                temp_candidate - baseline_outlet_temp
            ) < abs(best_temp - baseline_outlet_temp):
                best_temp = temp_candidate

    logging.debug(f"--- Optimal float temp found: {best_temp:.1f}°C ---")

    if not prediction_history:
        prediction_history.append(best_temp)
    else:
        last_smoothed = prediction_history[-1]
        alpha = 0.5
        smoothed_temp = alpha * best_temp + (1 - alpha) * last_smoothed
        prediction_history.append(smoothed_temp)

    logging.debug("--- Prediction Smoothing ---")
    history_formatted = [f"{t:.1f}" for t in prediction_history]
    logging.debug(f"  History: {history_formatted}")
    logging.debug(f"  Smoothed Temp: {prediction_history[-1]:.1f}°C")
    best_temp = prediction_history[-1]

    floor_temp, ceil_temp = np.floor(best_temp), np.ceil(best_temp)

    if floor_temp == ceil_temp:
        final_temp = best_temp
    else:
        temps_to_check = [floor_temp, ceil_temp]
        predictions = []

        for temp_candidate in temps_to_check:
            x_candidate = x_base.copy()
            x_candidate["outlet_temp"] = temp_candidate
            x_candidate["outlet_temp_sq"] = temp_candidate**2
            x_candidate["outlet_temp_cub"] = temp_candidate**3
            x_candidate["outlet_temp_change_from_last"] = (
                temp_candidate - last_outlet_temp
            )
            x_candidate["outlet_indoor_diff"] = temp_candidate - current_temp
            x_candidate["outdoor_temp_x_outlet_temp"] = (
                x_base["outdoor_temp"] * temp_candidate
            )

            try:
                predicted_delta = model.predict_one(x_candidate)
                predicted_indoor = current_temp + predicted_delta
                predictions.append((temp_candidate, predicted_indoor))
            except Exception:
                continue

        if not predictions:
            final_temp = round(best_temp)
        else:
            best_int_temp, min_int_diff = (
                predictions[0][0],
                abs(predictions[0][1] - target_temp),
            )
            for temp, indoor in predictions:
                diff = abs(indoor - target_temp)
                if diff < min_int_diff:
                    min_int_diff, best_int_temp = diff, temp
            final_temp = best_int_temp

            logging.debug("--- Smart Rounding ---")
            for temp, indoor in predictions:
                logging.debug(
                    f"  - Candidate {temp}°C -> "
                    f"Predicted: {indoor:.2f}°C "
                    f"(Diff: {abs(indoor - target_temp):.2f})"
                )
            logging.debug(f"  -> Chose: {final_temp}°C")

    return final_temp, confidence, prediction_history


def get_feature_importances(model: compose.Pipeline) -> Dict[str, float]:
    """
    Get feature importances from a River model by traversing the trees.
    This is more robust for young models than relying on the internal
    `feature_importances_` attribute.
    """
    regressor = model.steps.get("learn")
    if not regressor:
        logging.debug("Regressor not found for feature importance.")
        return {}

    logging.debug(
        "Traversing trees for feature importances."
    )

    total_importances: Dict[str, int] = defaultdict(int)
    for i, tree_model in enumerate(regressor):
        # The tree_model from AdaptiveRandomForestRegressor is the tree itself.
        if hasattr(tree_model, "_root"):

            def traverse(node):
                if node is None:
                    return

                # A node can be a SplitNode or a LearningNode.
                # Only SplitNodes have a 'feature' attribute.
                if hasattr(node, "feature") and node.feature is not None:
                    feature = node.feature
                    total_importances[feature] += 1

                # SplitNodes have a 'children' attribute which is a list.
                if hasattr(node, "children"):
                    for child in node.children:
                        traverse(child)

            traverse(tree_model._root)
        else:
            logging.debug(
                f"Tree {i} ({type(tree_model)}) has no _root attribute."
            )

    if not total_importances:
        logging.debug("No feature splits were found in any tree.")
        return {}

    logging.debug(f"Raw feature split counts: {dict(total_importances)}")

    total_splits = sum(total_importances.values())
    if total_splits > 0:
        return {
            f: count / total_splits for f, count in total_importances.items()
        }

    logging.debug("Total feature splits is 0, cannot normalize.")
    return {}
