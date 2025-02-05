from evaluate_models import load_all_models, load_all_thresholds, calculate_mcc
from read_data import required_columns, cf_test_data, cg_test_data
import json
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
import itertools


def load_evaluation_results(prefix, model_name):
    """Load the evaluation results from JSON."""
    file_path = f"./evaluation_results/{prefix}_{model_name}_evaluation.json"
    with open(file_path, "r") as f:
        evaluation_results = json.load(f)
    return evaluation_results


def predict_with_model(model, test_data):
    if "flight" in test_data.columns:
        X_test = test_data.drop(columns=["flight"])
    else:
        X_test = test_data.copy()
    predictions = model.predict_proba(X_test)[:, 1]
    return predictions


def weighted_ensemble_predictions(weights, models, X_train):
    """Compute weighted ensemble predictions."""
    glm_pred = predict_with_model(models["GLM"], X_train)
    lda_pred = predict_with_model(models["LDA"], X_train)
    nn_pred = predict_with_model(models["NN"], X_train)
    svm_pred = predict_with_model(models["SVM"], X_train)
    rf_pred = predict_with_model(models["RF"], X_train)

    weighted_sum = (
        weights[0] * glm_pred +
        weights[1] * lda_pred +
        weights[2] * nn_pred +
        weights[3] * svm_pred +
        weights[4] * rf_pred
    )

    return weighted_sum


def generate_weight_combinations():
    """Generate all valid weight combinations summing to 1 using 0.1 steps."""
    weight_values = np.arange(
        0.1, 1.1, 0.05)  # Possible values [0.1, 0.2, ..., 1.0]
    all_combinations = itertools.product(weight_values, repeat=5)
    valid_combinations = [w for w in all_combinations if round(
        sum(w), 1) == 1.0]  # Ensure sum == 1
    return valid_combinations


def find_best_weight(train_data, prefix):
    """Find the best ensemble weights using Grid Search based on MCC."""
    models = load_all_models(prefix)
    X_train = train_data.drop(columns=["flight"])
    Y_train = train_data["flight"].values  # Convert to NumPy array

    best_mcc = -1
    best_weights = None

    for weights in generate_weight_combinations():
        predictions = weighted_ensemble_predictions(weights, models, X_train)
        binary_predictions = (predictions > 0.5).astype(int)  # Apply threshold
        mcc_score = calculate_mcc(Y_train, binary_predictions)
        if mcc_score > best_mcc:
            best_mcc = mcc_score
            best_weights = weights
            print(
                f"Found new best weights for better mcc: {best_weights} -> {best_mcc}")

    # Save the best weights
    best_weights_dict = {
        "GLM": best_weights[0],
        "LDA": best_weights[1],
        "NN": best_weights[2],
        "SVM": best_weights[3],
        "RF": best_weights[4]
    }

    os.makedirs("ensemble_weights", exist_ok=True)
    file_path = f"ensemble_weights/{prefix}_ensemble_weights.json"

    with open(file_path, "w") as f:
        json.dump(best_weights_dict, f, indent=4)

    print(f"Optimized weights saved to {file_path}")
    return best_weights_dict


def load_best_weights(prefix):
    """Load the best ensemble weights from the ensemble_weights folder."""
    file_path = f"ensemble_weights/{prefix}_ensemble_weights.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            best_weights = json.load(f)
        return best_weights


def calculate_ensemble_score(test_data, prefix):
    """Calculate ensemble score using the best weights loaded from the file."""
    models = load_all_models(prefix)
    best_weights = load_best_weights(prefix)  # Load optimized weights

    # Generate predictions using each model
    glm_predictions = predict_with_model(models["GLM"], test_data)
    lda_predictions = predict_with_model(models["LDA"], test_data)
    nn_predictions = predict_with_model(models["NN"], test_data)
    svm_predictions = predict_with_model(models["SVM"], test_data)
    rf_predictions = predict_with_model(models["RF"], test_data)

    # Compute weighted ensemble score using best weights
    weighted_ensemble_score = (
        best_weights["GLM"] * glm_predictions +
        best_weights["LDA"] * lda_predictions +
        best_weights["NN"] * nn_predictions +
        best_weights["SVM"] * svm_predictions +
        best_weights["RF"] * rf_predictions
    )

    return weighted_ensemble_score


def cf_ensemble_model(input_df):
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T
    input_df = input_df.drop(columns=["flight"], errors="ignore")
    input_df = input_df[required_columns]
    ensemble_scores = calculate_ensemble_score(
        input_df, "cf")
    return ensemble_scores


def cg_ensemble_model(input_df):
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T
    input_df = input_df.drop(columns=["flight"], errors="ignore")
    input_df = input_df[required_columns]
    ensemble_scores = calculate_ensemble_score(
        input_df, "cg")
    return ensemble_scores


#find_best_weight(cf_test_data, "cf")

#find_best_weight(cg_test_data, "cg")
