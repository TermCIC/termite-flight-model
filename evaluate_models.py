import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
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
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Highest MCC: {highest_mcc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(actual_classes, final_classes))
    print("Classification Report:")
    print(classification_report(actual_classes, final_classes))
    return final_classes


def evaluate_model(model, test_data, target_column="flight"):
    """Evaluate a trained model with predictions and MCC."""
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    predictions = model.predict_proba(X_test)[:, 1]
    final_classes = calculate_threshold(predictions, test_data, target_column)
    mcc = calculate_mcc(final_classes, y_test)
    print(f"MCC: {mcc:.3f}")
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
        predictions, final_classes, mcc = evaluate_model(model, test_data)
        plot_results(test_data, final_classes, title=f"{prefix.upper()} {name} Predictions")


# Main evaluation function
def run():
    print("Evaluating CF dataset...")
    run_evaluation(cf_test_data, "cf")

    print("\nEvaluating CG dataset...")
    run_evaluation(cg_test_data, "cg")


if __name__ == "__main__":
    run()
