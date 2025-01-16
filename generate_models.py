import os
from read_data import *
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Ensure the models directory exists
os.makedirs("./models", exist_ok=True)

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
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
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


# Train Models
cf_glm = produce_glm(cf_train_data, "cf_glm")
cf_rf, _ = produce_rf(cf_train_data, "cf_rf")
cf_lda = produce_lda(cf_train_data, "cf_lda")
cf_svm, _ = produce_svm(cf_train_data, "cf_svm")
cf_nn, _ = produce_nn(cf_train_data, "cf_nn")

cg_glm = produce_glm(cg_train_data, "cg_glm")
cg_rf, _ = produce_rf(cg_train_data, "cg_rf")
cg_lda = produce_lda(cg_train_data, "cg_lda")
cg_svm, _ = produce_svm(cg_train_data, "cg_svm")
cg_nn, _ = produce_nn(cg_train_data, "cg_nn")
