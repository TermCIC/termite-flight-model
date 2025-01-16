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

# Ensure the models directory exists
os.makedirs("./models", exist_ok=True)

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
        mcc = matthews_corrcoef(actual_classes, predicted_classes)
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

# Model training functions
def train_and_save_model(model, train_data, treatment_name, target_column="flight"):
    """Train and save a machine learning model."""
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    model.fit(X_train, y_train)
    file_path = os.path.join("./models", f"{treatment_name}_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")
    return model


def train_with_grid_search(model, param_grid, train_data, treatment_name, target_column="flight"):
    """Train a model with hyperparameter tuning and save the best model."""
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    file_path = os.path.join("./models", f"{treatment_name}_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {file_path}")
    return best_model, grid_search.best_params_


# Logistic Regression
def produce_glm(train_data, treatment_name):
    model = LogisticRegression(solver="saga", max_iter=100000, random_state=123)
    return train_and_save_model(model, train_data, treatment_name)


# Random Forest
def produce_rf(train_data, treatment_name):
    model = RandomForestClassifier(random_state=123)
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]}
    return train_with_grid_search(model, param_grid, train_data, treatment_name)


# Linear Discriminant Analysis
def produce_lda(train_data, treatment_name):
    model = LinearDiscriminantAnalysis()
    return train_and_save_model(model, train_data, treatment_name)


# Support Vector Machine
def produce_svm(train_data, treatment_name):
    model = SVC(probability=True, random_state=123)
    param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf"]}
    return train_with_grid_search(model, param_grid, train_data, treatment_name)


# Neural Network
def produce_nn(train_data, treatment_name):
    model = MLPClassifier(max_iter=10000, random_state=123)
    param_grid = {
        "hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
    }
    return train_with_grid_search(model, param_grid, train_data, treatment_name)

# Neural Network
def produce_nn(train_data, test_data, treatment_name, target_column="flight"):
    """
    Train a Neural Network model with grid search for hyperparameter tuning.
    Debugging enabled to print test data and predictions.
    """
    print(f"Starting Neural Network training for {treatment_name}...")
    
    # Define the model and parameter grid
    model = MLPClassifier(max_iter=30000, random_state=123)
    param_grid = {
        "hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25)],
        "activation": ["tanh"],
        "solver": ["adam", "sgd"],
    }
    
    # Perform grid search for hyperparameter tuning
    best_model, best_params = train_with_grid_search(model, param_grid, train_data, treatment_name, target_column)
    print(f"Best Hyperparameters for {treatment_name}: {best_params}")
    
    # Test the model
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    predictions = best_model.predict_proba(X_test)[:, 1]  # Predicted probabilities
    
    # Find the best threshold and calculate MCC
    final_classes = calculate_threshold(predictions, test_data, target_column)
    mcc = matthews_corrcoef(y_test, final_classes)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")
    
    # Save the best model
    file_path = os.path.join("./models", f"{treatment_name}_model.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {file_path}")
    
    # Return the model, predictions, final classes, and MCC for further inspection
    return best_model, predictions, final_classes, mcc


# Example Usage
nn_model, predictions, final_classes, mcc = produce_nn(cf_train_data, cf_test_data, "cf_nn", target_column="flight")

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
