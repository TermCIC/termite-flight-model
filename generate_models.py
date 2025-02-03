import os
from read_data import *
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import json


# Ensure the models directory exists
os.makedirs("./models", exist_ok=True)


def display_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Display a confusion matrix with a heatmap.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - class_names: List of class names (optional).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_threshold(probabilities, train_data, treatment_name):
    best_threshold = 0
    highest_mcc = -1
    actual_classes = train_data["flight"]
    for threshold in [i / 1000 for i in range(1001)]:
        predicted_classes = predict_with_threshold(probabilities, threshold)
        mcc = matthews_corrcoef(actual_classes, predicted_classes)
        if mcc > highest_mcc:
            highest_mcc = mcc
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Highest MCC: {highest_mcc:.3f}")
    # Prepare results for JSON
    threshold_results = {
        "Best Threshold": best_threshold,
    }
    # Save results to a JSON file
    os.makedirs("./threshold_results", exist_ok=True)
    json_file = f"./threshold_results/{treatment_name}_threshold.json"
    with open(json_file, "w") as f:
        json.dump(threshold_results, f, indent=4)
    print(f"Saved threshold results to {json_file}")


def predict_with_threshold(probabilities, threshold):
    """Convert probabilities to binary predictions based on threshold."""
    return (probabilities > threshold).astype(int)


# Model training functions
def train_and_save_model(model, train_data, treatment_name):
    """Train and save a machine learning model."""
    X_train = train_data.drop(columns=["flight"])
    y_train = train_data["flight"]
    model.fit(X_train, y_train)
    file_path = os.path.join("./models", f"{treatment_name}_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")
    probabilities = model.predict_proba(X_train)[:, 1]
    calculate_threshold(probabilities, train_data, treatment_name)
    return model


def train_with_grid_search(model, param_grid, train_data, treatment_name):
    """Train a model with hyperparameter tuning and save the best model."""
    X_train = train_data.drop(columns=["flight"])
    y_train = train_data["flight"]
    grid_search = GridSearchCV(
        model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    file_path = os.path.join("./models", f"{treatment_name}_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {file_path}")
    probabilities = best_model.predict_proba(X_train)[:, 1]
    calculate_threshold(probabilities, train_data, treatment_name)
    return best_model, grid_search.best_params_


# Logistic Regression
def produce_glm(train_data, treatment_name):
    model = LogisticRegression(
        solver="saga", max_iter=100000, random_state=168)
    trained_model = train_and_save_model(model, train_data, treatment_name)
    return trained_model


# Random Forest
def produce_rf(train_data, treatment_name):
    model = RandomForestClassifier(random_state=168)
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}
    best_model, _ = train_with_grid_search(
        model, param_grid, train_data, treatment_name)
    return best_model


# Linear Discriminant Analysis (LDA)
def produce_lda(train_data, treatment_name):
    model = LinearDiscriminantAnalysis()
    trained_model = train_and_save_model(model, train_data, treatment_name)
    return trained_model


# Support Vector Machine (SVM)
def produce_svm(train_data, treatment_name):
    model = SVC(probability=True, random_state=168)
    param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf"]}
    best_model, _ = train_with_grid_search(
        model, param_grid, train_data, treatment_name)
    return best_model


# Neural Network
def produce_nn(train_data, treatment_name):
    model = MLPClassifier(max_iter=1000000, random_state=168)
    param_grid = {
        "hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25), (200,)],
        "activation": ["tanh", "relu"],
        "solver": ["adam", "sgd"],
    }
    best_model, _ = train_with_grid_search(
        model, param_grid, train_data, treatment_name)
    return best_model


# Train Models
def generate_models():
    produce_glm(cf_train_data, "cf_glm")
    produce_rf(cf_train_data, "cf_rf")
    produce_lda(cf_train_data, "cf_lda")
    produce_svm(cf_train_data, "cf_svm")
    produce_nn(cf_train_data, "cf_nn")

    produce_glm(cg_train_data, "cg_glm")
    produce_rf(cg_train_data, "cg_rf")
    produce_lda(cg_train_data, "cg_lda")
    produce_svm(cg_train_data, "cg_svm")
    produce_nn(cg_train_data, "cg_nn")


generate_models()
