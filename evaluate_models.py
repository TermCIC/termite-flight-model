import os
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    accuracy_score,
)
from read_data import cf_test_data, cg_test_data


# Function to load models from the models directory
def load_model(file_name):
    """Load a model from a pickle file."""
    file_path = os.path.join("./models", file_name)
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


# Helper functions
def calculate_mcc(predicted, actual):
    """Calculate Matthews Correlation Coefficient."""
    return matthews_corrcoef(actual, predicted)


def predict_with_threshold(probabilities, threshold):
    """Convert probabilities to binary predictions based on threshold."""
    return (probabilities > threshold).astype(int)


def calculate_threshold(predictions, test_data, target_column="flight"):
    """Find the best threshold for MCC."""
    best_threshold = 0
    highest_mcc = -1
    actual_classes = test_data[target_column]

    # Search for the best threshold
    for threshold in [i / 1000 for i in range(1001)]:
        predicted_classes = predict_with_threshold(predictions, threshold)
        mcc = calculate_mcc(predicted_classes, actual_classes)
        if mcc > highest_mcc:
            highest_mcc = mcc
            best_threshold = threshold

    final_classes = predict_with_threshold(predictions, best_threshold)
    return final_classes, best_threshold, highest_mcc


def evaluate_model(predictions, test_data, prefix, model_name, target_column="flight"):
    """Evaluate a trained model with predictions, MCC, and save results."""
    y_test = test_data[target_column]
    final_classes, best_threshold, mcc = calculate_threshold(predictions, test_data, target_column)

    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(y_test, final_classes)
    accuracy = accuracy_score(y_test, final_classes)

    # Prepare results for JSON
    evaluation_results = {
        "Accuracy": accuracy,
        "MCC": mcc,
        "Best Threshold": best_threshold,
        "Confusion Matrix": {
            "Actual 0": {"Predicted 0": int(cm[0, 0]), "Predicted 1": int(cm[0, 1])},
            "Actual 1": {"Predicted 0": int(cm[1, 0]), "Predicted 1": int(cm[1, 1])},
        },
    }

    # Save results to a JSON file
    os.makedirs("./evaluation_results", exist_ok=True)
    json_file = f"./evaluation_results/{prefix}_{model_name}_evaluation.json"
    with open(json_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Saved evaluation results to {json_file}")

    return predictions, final_classes, mcc


def plot_results(test_data, final_classes, title="Model Predictions"):
    """Plot prediction results."""
    plt.scatter(test_data["day"] / 30, final_classes, alpha=0.5)
    plt.xlabel("Month")
    plt.ylabel("Prediction of flight event")
    plt.title(title)
    plt.show()


def load_all_models(prefix):
    """Load all models for a specific dataset."""
    models = {
        "GLM": load_model(f"{prefix}_glm_model.pkl"),
        "RF": load_model(f"{prefix}_rf_model.pkl"),
        "LDA": load_model(f"{prefix}_lda_model.pkl"),
        "SVM": load_model(f"{prefix}_svm_model.pkl"),
        "NN": load_model(f"{prefix}_nn_model.pkl"),
    }
    return models


def run_evaluation(test_data, prefix):
    """Run evaluation for all models in the specified prefix."""
    models = load_all_models(prefix)
    for name, model in models.items():
        print(f"Evaluating {prefix} {name}...")
        predictions = model.predict_proba(X_test)[:, 1]
        final_classes, mcc = evaluate_model(predictions, test_data, prefix, name)
        plot_results(test_data, final_classes, title=f"{prefix.upper()} {name} Predictions")


# Main evaluation function
def run_evaluation():
    print("Evaluating CF dataset...")
    run_evaluation(cf_test_data, "cf")

    print("\nEvaluating CG dataset...")
    run_evaluation(cg_test_data, "cg")