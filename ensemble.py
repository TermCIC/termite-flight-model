from evaluate_models import *
import numpy as np

def load_evaluation_results(prefix, model_name):
    """Load the evaluation results from JSON."""
    file_path = f"./evaluation_results/{prefix}_{model_name}_evaluation.json"
    with open(file_path, "r") as f:
        evaluation_results = json.load(f)
    return evaluation_results


def predict_with_model(model, test_data, threshold, target_column="flight"):
    if target_column in test_data.columns:
        X_test = test_data.drop(columns=[target_column])
    else:
        X_test = test_data.copy()
    predictions = model.predict_proba(X_test)[:, 1]
    return (predictions > threshold).astype(int)



def calculate_ensemble_score(test_data, prefix, target_column="flight"):
    """Combine the predictions of five models to create an ensemble score."""
    # Load models
    reg_model = load_model(f"{prefix}_glm_model.pkl")
    lda_model = load_model(f"{prefix}_lda_model.pkl")
    nn_model = load_model(f"{prefix}_nn_model.pkl")
    svm_model = load_model(f"{prefix}_svm_model.pkl")
    rf_model = load_model(f"{prefix}_rf_model.pkl")

    # Load thresholds
    reg_threshold = load_evaluation_results(prefix, "glm").get("Best Threshold")
    lda_threshold = load_evaluation_results(prefix, "lda").get("Best Threshold")
    nn_threshold = load_evaluation_results(prefix, "nn").get("Best Threshold")
    svm_threshold = load_evaluation_results(prefix, "svm").get("Best Threshold")
    rf_threshold = load_evaluation_results(prefix, "rf").get("Best Threshold")

    # Generate predictions with thresholds
    reg_predictions = predict_with_model(reg_model, test_data, reg_threshold, target_column)
    lda_predictions = predict_with_model(lda_model, test_data, lda_threshold, target_column)
    nn_predictions = predict_with_model(nn_model, test_data, nn_threshold, target_column)
    svm_predictions = predict_with_model(svm_model, test_data, svm_threshold, target_column)
    rf_predictions = predict_with_model(rf_model, test_data, rf_threshold, target_column)

    # Calculate ensemble score
    raw_ensemble_score = (
        reg_predictions + lda_predictions + nn_predictions + svm_predictions + rf_predictions
    )

    # Apply normalization: max(raw_ensemble_score - 1, 0) / 4
    ensemble_score = np.maximum(raw_ensemble_score - 1, 0) / 4
    return ensemble_score