"""
This module encapsulates all machine learning-related logic.

It handles the creation, training, prediction, and persistence of the online
learning model. The core of this module is the `ModelWrapper` class, which
uses the River library for online machine learning.
"""
import logging
import pickle
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from river import compose, drift, metrics, preprocessing
from river.forest import ARFRegressor

from . import config
from .feature_builder import build_features_for_training, get_feature_names


def create_model() -> compose.Pipeline:
    """
    Builds and returns a new River-based online learning pipeline.

    The pipeline is composed of several stages:
    1.  **Feature Scaling**: A `StandardScaler` is applied to most numerical
        features to normalize their range, which helps the model learn more
        effectively. Features directly involved in the temperature search
        (like 'outlet_temp') are passed through without scaling to simplify
        the optimization process.
    2.  **Model**: An `ARFRegressor` (Adaptive Random Forest Regressor) is used.
        This is an online ensemble method that is well-suited for streaming
        data and can adapt to concept drift.
    """
    feature_names = get_feature_names()

    # We separate features because the 'outlet_temp' and its derivatives are
    # the variables we are searching over. Scaling them would complicate the
    # search process, as we would need to inverse-transform them constantly.
    # Other features are scaled to bring them to a similar magnitude, which
    # generally helps the model learn more effectively.
    unscaled_features = [
        "outlet_temp",
        "outlet_temp_sq",
        "outlet_temp_cub",
        "outlet_temp_change_from_last",
        "outlet_indoor_diff",
        "outdoor_temp_x_outlet_temp",
    ]

    scaled_features = [f for f in feature_names if f not in unscaled_features]

    # Create a sub-pipeline to apply standard scaling to the selected features.
    scaler = compose.Select(
        *scaled_features
    ) | preprocessing.StandardScaler()

    # Create a sub-pipeline that simply passes the unscaled features through.
    passthrough = compose.Select(*unscaled_features)

    # Combine the scaler and passthrough pipelines. The '+' operator creates a
    # TransformerUnion, which applies the transformations in parallel and
    # concatenates the results.
    preprocessor = scaler + passthrough

    # The final model pipeline.
    model_pipeline = compose.Pipeline(
        ("features", preprocessor),
        (
            "learn",
            ARFRegressor(
                n_models=10,  # The number of trees in the forest.
                seed=42,
                # Drift detectors allow the model to adapt to changes in the
                # data's underlying distribution (concept drift).
                drift_detector=drift.PageHinkley(),
                warning_detector=drift.ADWIN(),
            ),
        ),
    )
    return model_pipeline


def load_model() -> Tuple[compose.Pipeline, metrics.MAE, metrics.RMSE]:
    """
    Loads the persisted model and its associated metrics from disk.

    It attempts to load from the file specified in `config.MODEL_FILE`.
    - If the file is not found or is corrupted, it logs a warning and
      initializes a fresh model and new metric trackers.
    - It supports backward compatibility for older save formats.

    Returns:
        A tuple containing the model pipeline, MAE metric tracker, and
        RMSE metric tracker.
    """
    try:
        with open(config.MODEL_FILE, "rb") as f:
            saved_data = pickle.load(f)
            # New format: a dictionary containing model and metrics
            if isinstance(saved_data, dict):
                model = saved_data["model"]
                mae = saved_data.get("mae", metrics.MAE())
                rmse = saved_data.get("rmse", metrics.RMSE())
                logging.info(
                    "Successfully loaded model and metrics from %s",
                    config.MODEL_FILE,
                )
            else:
                # Handle backward compatibility with the old format where only
                # the model was saved.
                model = saved_data
                mae = metrics.MAE()
                rmse = metrics.RMSE()
                logging.info(
                    "Successfully loaded old format model from %s, creating new metrics.",
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
    """
    Saves the current state of the model and its metrics to a file.

    This is crucial for persistence, allowing the service to restart without
    losing all learned knowledge. The model and metrics are bundled into a
    dictionary and pickled.

    Args:
        model: The River pipeline model.
        mae: The Mean Absolute Error metric object.
        rmse: The Root Mean Squared Error metric object.
    """
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
    """
    Performs an initial "batch" training session on historical data.

    While the model is designed for online learning, this function gives it a
    head start by training on a recent window of data from InfluxDB. This
    helps the model converge to a reasonable state faster than starting from
    scratch with live data.

    Args:
        model: The River pipeline model to be trained.
        mae: The MAE metric tracker.
        rmse: The RMSE metric tracker.
        influx_service: The service for querying historical data.
    """
    logging.info("--- Initial Model Training ---")
    # Fetch historical data based on the configured lookback period.
    df = influx_service.get_training_data(
        lookback_hours=config.TRAINING_LOOKBACK_HOURS
    )
    if df.empty or len(df) < 240:  # Need at least 20 hours of data
        logging.warning("Not enough data for initial training. Skipping.")
        return

    features_list, labels_list = [], []

    # Iterate through the historical data to create training samples.
    # We start 12 steps in to ensure we have enough history for the first sample.
    for idx in range(12, len(df) - config.PREDICTION_HORIZON_STEPS):
        features = build_features_for_training(df, idx)
        if features is None:
            continue

        # The label is the change in indoor temperature over the prediction
        # horizon.
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

        # We let the scaler and the regressor learn online. We give the model
        # a head start (warm-up period) before tracking metrics, as the very
        # first predictions can be unstable while the model is still naive.
        logging.info("Performing initial model training...")
        regressor_warm_up_samples = 100
        for i, (features, label) in enumerate(zip(features_list, labels_list)):
            y_pred = model.predict_one(features)
            model.learn_one(features, label)

            # Start tracking metrics only after the warm-up period.
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
    prediction_history: list,
    outlet_history: list[float],
    error_target_vs_actual: float,
    outdoor_temp: float,
) -> tuple[float, float, list, float]:
    """
    The core optimization function to find the ideal heating outlet temperature.

    This function orchestrates a multi-step process to determine the best
    temperature setting:
    1.  **Confidence Check**: It first assesses the model's confidence by
        measuring the standard deviation of predictions from individual trees
        in the forest. If confidence is low, it falls back to a safe baseline.
    2.  **Search**: It performs a grid search over a range of possible outlet
        temperatures, using the model to predict the resulting indoor
        temperature change for each candidate.
    3.  **Smoothing**: The best temperature from the search is smoothed using
        an exponential moving average of recent predictions to prevent rapid,
        jarring changes in the setpoint.
    4.  **Smart Rounding**: Instead of a simple round, it checks both the floor
        and ceiling integer temperatures to see which one is predicted to get
        closer to the target, making a more intelligent final decision.

    Args:
        model: The trained River model.
        features: The input features for the current time step.
        current_temp: The current indoor temperature.
        target_temp: The desired indoor temperature.
        prediction_history: A list of recent smoothed predictions.
        outlet_history: A list of recent actual outlet temperatures.

    Returns:
        A tuple with the final outlet temperature, model confidence, and the
        updated prediction history.
    """
    logging.info("--- Finding Best Outlet Temp ---")
    logging.info(f"Target indoor temp: {target_temp:.1f}°C")
    best_temp, min_diff = outlet_history[-1], float("inf")

    x_base = features.to_dict(orient="records")[0]

    # --- Confidence Monitoring ---
    # The model's uncertainty is measured by `sigma`, the standard deviation
    # of predictions from the internal trees. A higher `sigma` means more
    # disagreement and thus lower confidence. This `sigma` is converted to a
    # `confidence` score between 0 (max uncertainty) and 1 (perfect
    # certainty).
    regressor = model.steps["learn"]
    tree_preds = [tree.predict_one(x_base) for tree in regressor]
    # Raw uncertainty (σ) as the standard deviation of tree predictions in °C
    sigma = float(np.std(tree_preds))

    # Normalize σ to a 0..1 confidence score where 1.0 == perfect agreement.
    # This mapping is monotonic: larger σ -> lower confidence.
    confidence = 1.0 / (1.0 + sigma)

    # If normalized confidence is below the configured threshold, fall back.
    if confidence < config.CONFIDENCE_THRESHOLD:
        logging.warning(
            "Model confidence low (σ=%.3f°C, confidence=%.3f < %.3f)."
            " Will still return ML proposal; state will indicate low confidence.",
            sigma,
            confidence,
            config.CONFIDENCE_THRESHOLD,
        )

    # --- Search for Optimal Temperature ---
    # Search across the configured absolute clamp range.
    min_search_temp = config.CLAMP_MIN_ABS
    max_search_temp = config.CLAMP_MAX_ABS
    step = 0.5

    if min_search_temp > max_search_temp:
        return outlet_history[-1], 0.0, prediction_history, 0.0

    last_outlet_temp = outlet_history[-1]

    # Iterate through candidate temperatures and predict the outcome.
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
        outdoor_temp = x_base["outdoor_temp"]
        x_candidate["outdoor_temp_x_outlet_temp"] = (
            outdoor_temp * temp_candidate
        )

        predicted_delta = model.predict_one(x_candidate)
        predicted_indoor = current_temp + predicted_delta

        logging.info(
            f"  - Test {temp_candidate:.1f}°C -> "
            f"Pred ΔT: {predicted_delta:.3f}°C, "
            f"Indoor: {predicted_indoor:.2f}°C"
        )
        diff = abs(predicted_indoor - target_temp)
        if diff < min_diff:
            min_diff, best_temp = diff, temp_candidate
        # Tie-breaking: if two temps are equally good, choose the one
        # closer to the last actual outlet temperature.
        elif diff == min_diff:
            is_closer = abs(temp_candidate - last_outlet_temp) < abs(
                best_temp - last_outlet_temp
            )
            if is_closer:
                best_temp = temp_candidate

    logging.info(f"--- Optimal float temp found: {best_temp:.1f}°C ---")

    # --- Prediction Smoothing ---
    # Apply exponential smoothing to the predictions to reduce volatility.
    if not prediction_history:
        prediction_history.append(best_temp)
    else:
        last_smoothed = prediction_history[-1]
        smoothed_temp = (
            config.SMOOTHING_ALPHA * best_temp
            + (1 - config.SMOOTHING_ALPHA) * last_smoothed
        )
        prediction_history.append(smoothed_temp)

    if len(prediction_history) > 50:
        del prediction_history[:-50]

    logging.debug("--- Prediction Smoothing ---")
    history_formatted = [f"{t:.1f}" for t in prediction_history]
    logging.debug(f"  History: {history_formatted}")
    logging.debug(f"  Smoothed Temp: {prediction_history[-1]:.1f}°C")
    best_temp = prediction_history[-1]

    # --- Dynamic Boost ---
    # Apply a corrective boost based on the current indoor temperature error.
    boost = 0.0
    if error_target_vs_actual > 0.1:  # If the room is too cold
        boost = min(
            error_target_vs_actual * 2.0, 5.0
        )  # Boost, capped at 5°C
    elif error_target_vs_actual < -0.1:  # If the room is too warm
        boost = max(
            error_target_vs_actual * 1.5, -5.0
        )  # Reduce, capped at -5°C

    # Disable boost if it's warm outside, as solar gain might be sufficient.
    if outdoor_temp > 15:
        boost = min(boost, 0)

    logging.info("--- Dynamic Boost ---")
    logging.info(f"  Model Suggested (Smoothed): {best_temp:.1f}°C")
    logging.info(f"  Dynamic Boost Applied: {boost:.2f}°C")
    boosted_temp = best_temp + boost
    logging.info(f"  Boosted Temp: {boosted_temp:.1f}°C")

    # Clamp the boosted temperature to the configured absolute range to
    # prevent extreme values.
    lower_bound = config.CLAMP_MIN_ABS
    upper_bound = config.CLAMP_MAX_ABS
    best_temp = np.clip(boosted_temp, lower_bound, upper_bound)
    logging.info(
        "  Clamped Boosted Temp (to %.1f-%.1f°C): %.1f°C",
        lower_bound,
        upper_bound,
        best_temp,
    )

    # --- Smart Rounding ---
    # Instead of simple rounding, check which integer (floor or ceil) gives a
    # better predicted outcome.
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
            outdoor_temp = x_base["outdoor_temp"]
            x_candidate["outdoor_temp_x_outlet_temp"] = (
                outdoor_temp * temp_candidate
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
            # Choose the integer temp that results in an indoor temp
            # closest to the target.
            best_int_temp, min_int_diff = predictions[0][0], abs(
                predictions[0][1] - target_temp
            )
            for temp, indoor in predictions:
                diff = abs(indoor - target_temp)
                if diff < min_int_diff:
                    min_int_diff, best_int_temp = diff, temp
            final_temp = best_int_temp

            logging.info("--- Smart Rounding ---")
            for temp, indoor in predictions:
                logging.info(
                    f"  - Candidate {temp}°C -> "
                    f"Predicted: {indoor:.2f}°C "
                    f"(Diff: {abs(indoor - target_temp):.2f})"
                )
            logging.info(f"  -> Chose: {final_temp}°C")

    return final_temp, confidence, prediction_history, sigma


def get_feature_importances(model: compose.Pipeline) -> Dict[str, float]:
    """
    Calculates and returns the feature importances from the model.

    For the tree-based ensemble model, importance is determined by how
    frequently a feature is used to make a split in the decision trees. This
    provides insight into what factors the model considers most influential.

    This function ensures that all possible features are included in the
    output, even if their importance is zero.

    Args:
        model: The trained River pipeline.

    Returns:
        A dictionary mapping every feature name to its normalized
        importance score.
    """
    # Initialize a dictionary with all feature names set to 0.0 importance.
    # This ensures that even unused features are included in the final
    # output.
    all_feature_names = get_feature_names()
    feature_importances = {name: 0.0 for name in all_feature_names}

    regressor = model.steps.get("learn")
    if not regressor:
        logging.debug("Regressor not found for feature importance.")
        return feature_importances

    logging.debug("Traversing trees for feature importances.")

    total_importances: Dict[str, int] = defaultdict(int)
    # The regressor is an ensemble of decision trees.
    for i, tree_model in enumerate(regressor):
        # The tree_model from ARFRegressor is the tree itself.
        if hasattr(tree_model, "_root"):

            # Recursively traverse the tree from the root.
            def traverse(node):
                if node is None:
                    return

                # A node can be a SplitNode or a LearningNode (leaf).
                # Only SplitNodes have a 'feature' attribute.
                if hasattr(node, "feature") and node.feature is not None:
                    feature = node.feature
                    # Increment the count for the feature used in this split.
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
        return feature_importances

    logging.debug(f"Raw feature split counts: {dict(total_importances)}")

    # Normalize the counts to get a percentage-like importance score.
    total_splits = sum(total_importances.values())
    if total_splits > 0:
        # Update the dictionary with the calculated importances.
        for feature, count in total_importances.items():
            if feature in feature_importances:
                feature_importances[feature] = count / total_splits
        return feature_importances

    logging.debug("Total feature splits is 0, cannot normalize.")
    return feature_importances
