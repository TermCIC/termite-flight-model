from read_data import *
from generate_models import produce_nn
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import pickle


# Function to load a model from a pickle file
def load_model(file_name):
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# Load models for the cf dataset
cf_glm = load_model("cf_glm_model.pkl")
cf_rf = load_model("cf_rf_model.pkl")
cf_lda = load_model("cf_lda_model.pkl")
cf_svm = load_model("cf_svm_model.pkl")
cf_nn = load_model("cf_nn_model.pkl")

# Load models for the cg dataset
cg_glm = load_model("cg_glm_model.pkl")
cg_rf = load_model("cg_rf_model.pkl")
cg_lda = load_model("cg_lda_model.pkl")
cg_svm = load_model("cg_svm_model.pkl")
cg_nn = load_model("cg_nn_model.pkl")


# Helper functions
def calculate_mcc(predicted, actual):
    """Calculate Matthews Correlation Coefficient."""
    return matthews_corrcoef(actual, predicted)


def predict_with_threshold(probabilities, threshold):
    """Convert probabilities to binary predictions based on threshold."""
    return (probabilities > threshold).astype(int)


def calculate_threshold(predictions, test_data, target_column="flight", binary=False):
    """Find the best threshold for MCC."""
    best_threshold = 0
    highest_mcc = -1
    actual_classes = test_data[target_column]

    if not binary:
        for threshold in [i / 1000 for i in range(1001)]:
            predicted_classes = predict_with_threshold(predictions, threshold)
            mcc = calculate_mcc(predicted_classes, actual_classes)
            if mcc > highest_mcc:
                highest_mcc = mcc
                best_threshold = threshold
    else:
        best_threshold = 0.5  # Default threshold for binary probabilities

    final_classes = predict_with_threshold(predictions, best_threshold)
    print(f"Best threshold: {best_threshold}")
    print(f"Highest MCC: {highest_mcc}")
    print("Confusion Matrix:")
    print(confusion_matrix(actual_classes, final_classes))
    return final_classes


def evaluate_model(model, test_data, target_column="flight", binary=False):
    """Evaluate a trained model with predictions and MCC."""
    predictions = model.predict_proba(test_data.drop(columns=[target_column]))[:, 1]
    final_classes = calculate_threshold(predictions, test_data, target_column, binary)
    mcc = calculate_mcc(final_classes, test_data[target_column])
    print(f"MCC: {mcc}")
    return predictions, final_classes, mcc


def plot_results(test_data, final_classes, title="Model Predictions"):
    """Plot prediction results."""
    plt.scatter(test_data["day"] / 30, final_classes, alpha=0.5)
    plt.xlabel("Month")
    plt.ylabel("Prediction of flight event")
    plt.title(title)
    plt.show()


def run():
    # Evaluate models for the cf dataset
    for name, model in [("GLM", cf_glm), ("RF", cf_rf), ("LDA", cf_lda), ("SVM", cf_svm), ("NN", cf_nn)]:
        predictions, final_classes, mcc = evaluate_model(model, cf_test_data)
        plot_results(cf_test_data, final_classes, title=f"CF {name} Predictions")

    # Evaluate models for the cg dataset
    for name, model in [("GLM", cg_glm), ("RF", cg_rf), ("LDA", cg_lda), ("SVM", cg_svm), ("NN", cg_nn)]:
        predictions, final_classes, mcc = evaluate_model(model, cg_test_data)
        plot_results(cg_test_data, final_classes, title=f"CG {name} Predictions")


cf_nn = produce_nn(cf_train_data, "cf_nn")
test_data = cf_test_data
test_model = cf_nn
predictions, final_classes, mcc = evaluate_model(test_model, test_data)
plot_results(test_data, final_classes, title=f"Predictions")
