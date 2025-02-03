from generate_ensemble_models import cf_ensemble_model, cg_ensemble_model
from evaluate_models import evaluate_model
from read_data import cf_test_data, cg_test_data
import os
import json
import pandas as pd

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)
evaluation_results_folder = "./evaluation_results"
threshold_results_folder = "./threshold_results"

cf_ensemble = cf_ensemble_model(cf_test_data)
cg_ensemble = cg_ensemble_model(cg_test_data)


def evaluate_ensemble_scores(ensemble_scores, test_data, prefix):
    """Evaluate ensemble scores at specific thresholds."""
    thresholds = [0.2, 0.4, 0.6, 0.8, 1]
    for threshold in thresholds:
        # Apply the threshold to generate predictions
        probabilities = (ensemble_scores >= threshold).astype(int)
        model_name = f"ensemble_score_{threshold:.2f}"
        # Evaluate using the existing evaluate_model function
        evaluation_results = evaluate_model(
            probabilities, threshold, test_data)
        # Save results to a JSON file
        os.makedirs("./evaluation_results", exist_ok=True)
        json_file = f"./evaluation_results/{prefix}_{model_name}_evaluation.json"
        with open(json_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Saved evaluation results to {json_file}")


def collect_evaluation_results(output_csv="./output/evaluation_results_summary.csv"):
    """Collect data from JSON files in evaluation_results and organize them into a table."""
    organized_data = []

    # Iterate through all JSON files in the folder
    for filename in os.listdir(evaluation_results_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(evaluation_results_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract metrics from the JSON file
            accuracy = data.get("Accuracy")
            mcc = data.get("MCC")
            confusion_matrix = data.get("Confusion Matrix")
            actual_0_pred_0 = confusion_matrix["Actual 0"]["Predicted 0"]
            actual_0_pred_1 = confusion_matrix["Actual 0"]["Predicted 1"]
            actual_1_pred_0 = confusion_matrix["Actual 1"]["Predicted 0"]
            actual_1_pred_1 = confusion_matrix["Actual 1"]["Predicted 1"]

            # Extract the model name from the filename
            model_name = filename.replace(".json", "")

            # Append the data as a dictionary
            organized_data.append({
                "Model Name": model_name,
                "Accuracy": accuracy,
                "MCC": mcc,
                "Actual 0 Predicted 0": actual_0_pred_0,
                "Actual 0 Predicted 1": actual_0_pred_1,
                "Actual 1 Predicted 0": actual_1_pred_0,
                "Actual 1 Predicted 1": actual_1_pred_1
            })

    # Create a DataFrame and save it as a CSV
    df = pd.DataFrame(organized_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved organized evaluation results to {output_csv}")


# Evaluate for CF dataset
evaluate_ensemble_scores(cf_ensemble, cf_test_data, prefix="cf")

# Evaluate for CG dataset
evaluate_ensemble_scores(cg_ensemble, cg_test_data, prefix="cg")

# Run the function to collect results
collect_evaluation_results()
