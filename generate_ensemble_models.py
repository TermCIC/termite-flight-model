from evaluate_models import load_all_models, load_all_thresholds
from read_data import required_columns
import json
import pandas as pd


def load_evaluation_results(prefix, model_name):
    """Load the evaluation results from JSON."""
    file_path = f"./evaluation_results/{prefix}_{model_name}_evaluation.json"
    with open(file_path, "r") as f:
        evaluation_results = json.load(f)
    return evaluation_results


def predict_with_model(model, test_data, threshold):
    if "flight" in test_data.columns:
        X_test = test_data.drop(columns=["flight"])
    else:
        X_test = test_data.copy()
    predictions = model.predict_proba(X_test)[:, 1]
    return (predictions > threshold).astype(int)


def calculate_ensemble_score(test_data, prefix, target_column="flight"):
    """Combine the predictions of five models to create an ensemble score."""
    # Load all models and thresholds using the helper functions
    models = load_all_models(prefix)
    thresholds = load_all_thresholds(prefix)

    # Generate predictions using each model and its threshold.
    glm_predictions = predict_with_model(
        models["GLM"], test_data, thresholds["GLM"])
    lda_predictions = predict_with_model(
        models["LDA"], test_data, thresholds["LDA"])
    nn_predictions = predict_with_model(
        models["NN"],  test_data, thresholds["NN"])
    svm_predictions = predict_with_model(
        models["SVM"], test_data, thresholds["SVM"])
    rf_predictions = predict_with_model(
        models["RF"],  test_data, thresholds["RF"])

    # Calculate ensemble score as the average of the five models' predictions
    raw_ensemble_score = (
        glm_predictions +
        lda_predictions +
        nn_predictions +
        svm_predictions +
        rf_predictions
    )

    ensemble_score = raw_ensemble_score / 5
    return ensemble_score


def cf_ensemble_model(input_df):
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T
    input_df = input_df.drop(columns=["flight"], errors="ignore")
    input_df = input_df[required_columns]
    ensemble_scores = calculate_ensemble_score(
        input_df, "cf", target_column="flight")
    return ensemble_scores


def cg_ensemble_model(input_df):
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame().T
    input_df = input_df.drop(columns=["flight"], errors="ignore")
    input_df = input_df[required_columns]
    ensemble_scores = calculate_ensemble_score(
        input_df, "cg", target_column="flight")
    return ensemble_scores
