from generate_ensemble_models import cf_ensemble_model, cg_ensemble_model
from read_data import north_data, middle_west_data, south_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def mean_ensemble_score(score_series, test_data):
    score_series = np.where(score_series >= 0.5, score_series, 0)
    df = pd.DataFrame({
        "day": test_data["day"],
        "score": score_series
    })
    df["score"] = df["score"].fillna(0)
    mean_scores = df.groupby("day")["score"].mean()
    # Reindex to include every day from 1 to 365, filling missing days with 0.
    mean_scores = mean_scores.reindex(range(1, 366), fill_value=0)
    return mean_scores


def calculate_interaction_score(ensemble_A, ensemble_B):
    ensemble_A = np.where(ensemble_A >= 0.5, ensemble_A, 0)
    ensemble_B = np.where(ensemble_B >= 0.5, ensemble_B, 0)
    # Multiply the two series elementwise
    product = ensemble_A * ensemble_B
    # Replace any NaN or inf values with 0
    product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)
    # Return the square root of the cleaned product
    return np.sqrt(product)


# North region
cf_north_ensemble = mean_ensemble_score(
    cf_ensemble_model(north_data), north_data)
cg_north_ensemble = mean_ensemble_score(
    cg_ensemble_model(north_data), north_data)
north_interaction_score = mean_ensemble_score(calculate_interaction_score(
    cf_ensemble_model(north_data), cg_ensemble_model(north_data)), north_data)

print(sum(north_interaction_score))

# Middle West region
cf_middle_west_ensemble = mean_ensemble_score(
    cf_ensemble_model(middle_west_data), middle_west_data)
cg_middle_west_ensemble = mean_ensemble_score(
    cg_ensemble_model(middle_west_data), middle_west_data)
middle_west_interaction_score = mean_ensemble_score(calculate_interaction_score(
    cf_ensemble_model(middle_west_data), cg_ensemble_model(middle_west_data)), middle_west_data)

print(sum(middle_west_interaction_score))

# South region
cf_south_ensemble = mean_ensemble_score(
    cf_ensemble_model(south_data), south_data)
cg_south_ensemble = mean_ensemble_score(
    cg_ensemble_model(south_data), south_data)
south_interaction_score = mean_ensemble_score(calculate_interaction_score(
    cf_ensemble_model(south_data), cg_ensemble_model(south_data)), south_data)
print(sum(south_interaction_score))


def plot_interactions_line(
    ensemble_score_species_A,
    ensemble_score_species_B,
    ensemble_score_interaction,
    title="",
    output_filename=None,
    smooth_window=25  # window size for moving average smoothing
):
    # Define the month ranges for a leap year (1-366 days)
    month_ranges = {
        "Jan": (1, 31),
        "Feb": (32, 60),
        "Mar": (61, 91),
        "Apr": (92, 121),
        "May": (122, 152),
        "Jun": (153, 182),
        "Jul": (183, 213),
        "Aug": (214, 244),
        "Sep": (245, 274),
        "Oct": (275, 305),
        "Nov": (306, 335),
        "Dec": (336, 366)
    }

    # Create an x-axis: days 1 through the length of the ensemble series
    days = np.arange(1, len(ensemble_score_species_A) + 1)

    # Create a DataFrame for plotting
    score_dist_data = pd.DataFrame({
        "day": days,
        "Coptotermes formosanus": ensemble_score_species_A,
        "Coptotermes gestroi": ensemble_score_species_B,
        "Interaction": ensemble_score_interaction,
    })

    # Reset index to ensure "day" is only a column
    score_dist_data = score_dist_data.reset_index(drop=True)

    # Replace any NaN values with 0 in the score columns
    for col in ["Coptotermes formosanus", "Coptotermes gestroi", "Interaction"]:
        score_dist_data[col] = score_dist_data[col].fillna(0)

    # Sort the DataFrame by day (should already be sorted)
    score_dist_data = score_dist_data.sort_values("day")

    # Create smoothed versions using a rolling (moving) average
    score_dist_data["Coptotermes formosanus_smooth"] = score_dist_data["Coptotermes formosanus"].rolling(
        window=smooth_window, center=True, min_periods=1).mean()
    score_dist_data["Coptotermes gestroi_smooth"] = score_dist_data["Coptotermes gestroi"].rolling(
        window=smooth_window, center=True, min_periods=1).mean()
    score_dist_data["Interaction_smooth"] = score_dist_data["Interaction"].rolling(
        window=smooth_window, center=True, min_periods=1).mean()

    # Set up the font properties using Arial
    arial_font = fm.FontProperties(family="Arial", size=14)
    arial_italic = fm.FontProperties(family="Arial", style="italic", size=14)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot each variable's smoothed version as a line plot
    ax.plot(
        score_dist_data["day"],
        score_dist_data["Coptotermes formosanus_smooth"],
        label="Coptotermes formosanus",
        alpha=0.8, linewidth=2
    )
    ax.plot(
        score_dist_data["day"],
        score_dist_data["Coptotermes gestroi_smooth"],
        label="Coptotermes gestroi",
        alpha=0.8, linewidth=2
    )
    ax.plot(
        score_dist_data["day"],
        score_dist_data["Interaction_smooth"],
        label="Interaction",
        alpha=0.8, linewidth=2, linestyle="--"
    )

    # Set labels, limits, and grid with Arial font
    # ax.set_xlabel("Day of Year", fontsize=14, fontname="Arial")
    ax.set_ylabel("Score", fontsize=14, fontname="Arial")
    ax.set_xlim(1, 366)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    legend = ax.legend(title="", fontsize=14, title_fontsize=14)
    for text in legend.get_texts():
        # For non-italic legend texts, apply Arial (italic for non-Interaction)
        if text.get_text() != "Interaction":
            text.set_fontproperties(arial_italic)
        else:
            text.set_fontproperties(arial_font)
    ax.tick_params(axis='both', labelsize=14)
    # For tick labels, you can set the font globally:
    plt.setp(ax.get_xticklabels(), fontname="Arial")
    plt.setp(ax.get_yticklabels(), fontname="Arial")

    # Create a secondary x-axis for month labels
    ax_months = ax.twiny()
    ax_months.set_xlim(ax.get_xlim())
    # Compute midpoints for each month based on the month_ranges
    ax_months.set_xticks([(start + end) // 2 for start,
                         end in month_ranges.values()])
    ax_months.set_xticklabels(
        list(month_ranges.keys()), fontsize=14, fontname="Arial")
    ax_months.tick_params(axis='both', labelsize=14)
    ax_months.xaxis.set_ticks_position("bottom")
    ax_months.spines["bottom"].set_position(("outward", 24))
    plt.setp(ax_months.get_xticklabels(), fontname="Arial")

    plt.title(title, fontsize=16, fontname="Arial")
    plt.tight_layout()

    # Save the plot to a file if an output filename is provided; otherwise, display it
    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved figure to {output_filename}")
    plt.show()


# Plot
plot_interactions_line(
    cf_north_ensemble,
    cg_north_ensemble,
    north_interaction_score,
    output_filename="output/north_ensemble_score.png"
)

plot_interactions_line(
    cf_middle_west_ensemble,
    cg_middle_west_ensemble,
    middle_west_interaction_score,
    output_filename="output/middle_west_ensemble_score.png"
)

plot_interactions_line(
    cf_south_ensemble,
    cg_south_ensemble,
    south_interaction_score,
    output_filename="output/south_ensemble_score.png"
)
