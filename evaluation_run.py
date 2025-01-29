from ensemble import *
from visualization import *

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

cf_ensemble = calculate_ensemble_score(
    cf_test_data, "cf", target_column="flight")
cg_ensemble = calculate_ensemble_score(
    cg_test_data, "cg", target_column="flight")

north_esemble = calculate_ensemble_score(
    north_data, "cg", target_column="flight"
)

def evaluate_ensemble_scores(ensemble_scores, test_data, prefix, target_column="flight"):
    """Evaluate ensemble scores at specific thresholds."""
    thresholds = [0, 0.25, 0.5, 0.75, 1]

    for threshold in thresholds:
        # Apply the threshold to generate predictions
        predictions = (ensemble_scores >= threshold).astype(int)
        model_name = f"ensemble_score_{threshold:.2f}"

        # Evaluate using the existing evaluate_model function
        evaluate_model(predictions, test_data, prefix,
                       model_name, target_column)


# Evaluate for CF dataset
evaluate_ensemble_scores(cf_ensemble, cf_test_data, prefix="cf")

# Evaluate for CG dataset
evaluate_ensemble_scores(cg_ensemble, cg_test_data, prefix="cg")

# Plot for CF dataset
plot_flight_frequency_density(
    cf_test_data, output_filename="output/cf_flight_density.png")
plot_ensemble_score_density_points(
    cf_ensemble, cf_test_data, output_filename="output/cf_ensemble_score.png")

# Plot for CG dataset
plot_flight_frequency_density(
    cg_test_data, output_filename="output/cg_flight_density.png")
plot_ensemble_score_density_points(
    cg_ensemble, cg_test_data, output_filename="output/cg_ensemble_score.png")

# Run the function to collect results
collect_evaluation_results()
