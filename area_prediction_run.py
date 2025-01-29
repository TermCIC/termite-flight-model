from ensemble import *
from visualization import *

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Function to apply thresholding (reduce 0.25, but keep minimum at 0)
def filter_ensemble_score(score_series):
    threshold = 0.5
    score_series = pd.Series(score_series)
    return score_series.where(score_series >= threshold, 0)

def calculate_interaction_score(ensemble_A, ensemble_B):
    return np.sqrt(ensemble_A * ensemble_B)

# North region
cf_north_ensemble = filter_ensemble_score(calculate_ensemble_score(north_data, "cf", target_column="flight"))
cg_north_ensemble = filter_ensemble_score(calculate_ensemble_score(north_data, "cg", target_column="flight"))
north_interaction_score = calculate_interaction_score(cf_north_ensemble, cg_north_ensemble)
print(sum(north_interaction_score))

# Middle West region
cf_middle_west_ensemble = filter_ensemble_score(calculate_ensemble_score(middle_west_data, "cf", target_column="flight"))
cg_middle_west_ensemble = filter_ensemble_score(calculate_ensemble_score(middle_west_data, "cg", target_column="flight"))
middle_west_interaction_score = calculate_interaction_score(cf_middle_west_ensemble, cg_middle_west_ensemble)
print(sum(middle_west_interaction_score))

# South region
cf_south_ensemble = filter_ensemble_score(calculate_ensemble_score(south_data, "cf", target_column="flight"))
cg_south_ensemble = filter_ensemble_score(calculate_ensemble_score(south_data, "cg", target_column="flight"))
south_interaction_score = calculate_interaction_score(cf_south_ensemble, cg_south_ensemble)
print(sum(south_interaction_score))

# South region
cf_south_ensemble = filter_ensemble_score(calculate_ensemble_score(south_data, "cf", target_column="flight"))
cg_south_ensemble = filter_ensemble_score(calculate_ensemble_score(south_data, "cg", target_column="flight"))
south_interaction_score = calculate_interaction_score(cf_south_ensemble, cg_south_ensemble)
print(sum(south_interaction_score))

# Combine all ensemble scores
total_cf_ensemble = filter_ensemble_score(cf_north_ensemble + cf_middle_west_ensemble + cf_south_ensemble)
total_cg_ensemble = filter_ensemble_score(cg_north_ensemble + cg_middle_west_ensemble + cg_south_ensemble)

# Calculate the total interaction score
total_interaction_score = calculate_interaction_score(total_cf_ensemble, total_cg_ensemble)
print("Total interaction score sum:", sum(total_interaction_score))

# Plot
plot_interactions_density(
    cf_north_ensemble,
    cg_north_ensemble,
    north_interaction_score,
    north_data,
    output_filename="output/north_ensemble_score.png"
    )

plot_interactions_density(
    cf_middle_west_ensemble,
    cg_middle_west_ensemble,
    middle_west_interaction_score,
    middle_west_data,
    output_filename="output/middle_west_ensemble_score.png"
    )

plot_interactions_density(
    cf_south_ensemble,
    cg_south_ensemble,
    south_interaction_score,
    south_data,
    output_filename="output/south_ensemble_score.png"
    )

# Concatenate data while resetting the index to avoid duplicate labels
combined_data = pd.concat([north_data, middle_west_data, south_data], ignore_index=True)

plot_interactions_density(
    total_cf_ensemble,
    total_cg_ensemble,
    total_interaction_score,
    combined_data,  # Combine data from all regions
    output_filename="output/total_ensemble_score.png"
)