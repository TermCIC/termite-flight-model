import os
import pickle
import json
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    accuracy_score,
)
from read_data import cf_train_data, cg_train_data, cf_test_data, cg_test_data, north_data, middle_west_data, south_data


# Function to load models from the models directory
def load_model(file_name):
    """Load a model from a pickle file."""
    file_path = os.path.join("./models", file_name)
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

# Function to load model thresholds


def load_threshold(file_name):
    """Load a model from a pickle file."""
    file_path = os.path.join("./threshold_results", file_name)
    with open(file_path, "r") as f:
        threshold_result = json.load(f)
    threshold = threshold_result.get("Best Threshold")
    return threshold


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


def load_all_thresholds(prefix):
    """Load all thresholds for a specific dataset."""
    thresholds = {
        "GLM": load_threshold(f"{prefix}_glm_threshold.json"),
        "RF": load_threshold(f"{prefix}_rf_threshold.json"),
        "LDA": load_threshold(f"{prefix}_lda_threshold.json"),
        "SVM": load_threshold(f"{prefix}_svm_threshold.json"),
        "NN": load_threshold(f"{prefix}_nn_threshold.json"),
    }
    return thresholds


# Helper functions
def calculate_mcc(predicted, actual):
    """Calculate Matthews Correlation Coefficient."""
    return matthews_corrcoef(actual, predicted)


def predict_with_threshold(probabilities, threshold):
    """Convert probabilities to binary predictions based on threshold."""
    return (probabilities > threshold).astype(int)


def evaluate_model(probabilities, test_data):
    """Evaluate a trained model with predictions, MCC, and save results."""
    y_test = test_data["flight"]
    final_classes = predict_with_threshold(probabilities, 0.5)
    mcc = calculate_mcc(final_classes, y_test)
    # Calculate confusion matrix and accuracy
    cm = confusion_matrix(y_test, final_classes)
    accuracy = accuracy_score(y_test, final_classes)
    # Prepare results for JSON
    evaluation_results = {
        "Accuracy": accuracy,
        "MCC": mcc,
        "Confusion Matrix": {
            "Actual 0": {"Predicted 0": int(cm[0, 0]), "Predicted 1": int(cm[0, 1])},
            "Actual 1": {"Predicted 0": int(cm[1, 0]), "Predicted 1": int(cm[1, 1])},
        },
    }
    return evaluation_results


def run_evaluation(test_data, prefix):
    """Run evaluation for all models in the specified prefix."""
    models = load_all_models(prefix)
    for name, model in models.items():
        print(f"Evaluating {prefix} {name}...")
        X_test = test_data.drop(columns=["flight"])
        probabilities = model.predict_proba(X_test)[:, 1]
        evaluation_results = evaluate_model(
            probabilities, test_data)
        # Save results to a JSON file
        os.makedirs("./evaluation_results", exist_ok=True)
        json_file = f"./evaluation_results/{prefix}_{name}_evaluation.json"
        with open(json_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Saved evaluation results to {json_file}")


# Main evaluation function
def run_all_evaluation():
    print("Evaluating CF dataset...")
    run_evaluation(cf_test_data, "cf")

    print("\nEvaluating CG dataset...")
    run_evaluation(cg_test_data, "cg")


run_all_evaluation()
