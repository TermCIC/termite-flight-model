from ensemble import *
from visualization import *

cf_ensemble = calculate_ensemble_score(
    cf_test_data, "cf", target_column="flight")
cg_ensemble = calculate_ensemble_score(
    cg_test_data, "cg", target_column="flight")


# Ensure the output directory exists
os.makedirs("output", exist_ok=True)


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
