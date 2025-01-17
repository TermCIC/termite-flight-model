import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Set global font to Arial
plt.rcParams["font.family"] = "Arial"


def plot_ensemble_score_by_day_with_months(ensemble_score, test_data, title="Ensemble Score by Day", target_column="flight", output_filename=None):
    """Prepare data and plot ensemble score by day with an additional bar for months."""
    # Define the month ranges for a non-leap year
    month_ranges = {
        "Jan": (1, 31),
        "Feb": (32, 59),
        "Mar": (60, 90),
        "Apr": (91, 120),
        "May": (121, 151),
        "Jun": (152, 181),
        "Jul": (182, 212),
        "Aug": (213, 243),
        "Sep": (244, 273),
        "Oct": (274, 304),
        "Nov": (305, 334),
        "Dec": (335, 365)
    }

    # Create a DataFrame for plotting
    score_dist_data = pd.DataFrame({
        "ensemble_score": ensemble_score,
        # Adjust to match the R code
        "flight": test_data[target_column].astype(int) - 1,
        "day": test_data["day"]
    })

    # Main figure and scatter plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(
        score_dist_data["day"],
        score_dist_data["ensemble_score"],
        alpha=0.01,
        c="black"  # Set points to black
    )
    # ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Ensemble Score", fontsize=12)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(1, 365)
    ax.set_xticks([])  # Manually set y-axis ticks
    ax.set_ylim(-0.05, 1.05)  # Assuming ensemble score is normalized to [0, 1]
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])  # Manually set y-axis ticks
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Flight", fontsize=10, title_fontsize=10)

    # Create a secondary axis below for the month ranges
    ax_months = ax.twiny()  # Secondary x-axis
    ax_months.set_xlim(ax.get_xlim())  # Align with the main x-axis
    ax_months.set_xticks(
        # Midpoint of each month range
        [(start + end) // 2 for start, end in month_ranges.values()]
    )
    ax_months.set_xticklabels(list(month_ranges.keys()), fontsize=12)
    ax_months.xaxis.set_ticks_position("bottom")  # Place ticks at the bottom
    ax_months.spines["bottom"].set_position(("outward", 0))  # Offset the axis
    # ax_months.set_xlabel("Month", fontsize=14)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved figure to {output_filename}")
    plt.show()


def plot_flight_frequency_density(test_data, title="Kernel Density Plot of Flight Frequency", target_column="flight", output_filename=None):
    """Generate a kernel density plot for the frequency of flight events in the dataset."""
    # Ensure the target column is numeric
    test_data[target_column] = test_data[target_column].astype(int)
    test_data["Flight"] = test_data[target_column]

    # Main figure and density plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Kernel Density Plot
    sns.kdeplot(
        data=test_data,
        x="day",
        hue="Flight",  # Use the new column for the legend
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={0: "blue", 1: "red"},
        ax=ax
    )

    # Set labels and limits
    ax.set_xlabel("")
    # ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.set_xlim(1, 365)
    ax.set_xticks([])  # Remove x-ticks for similar style
    ax.set_ylim(0, 0.05)  # Allow y-axis to adjust automatically
    ax.grid(True, linestyle="--", alpha=0.6)

    # Create a secondary axis below for the month ranges
    month_ranges = {
        "Jan": (1, 31),
        "Feb": (32, 59),
        "Mar": (60, 90),
        "Apr": (91, 120),
        "May": (121, 151),
        "Jun": (152, 181),
        "Jul": (182, 212),
        "Aug": (213, 243),
        "Sep": (244, 273),
        "Oct": (274, 304),
        "Nov": (305, 334),
        "Dec": (335, 365)
    }
    ax_months = ax.twiny()
    ax_months.set_xlim(ax.get_xlim())
    ax_months.set_xticks(
        [(start + end) // 2 for start, end in month_ranges.values()]
    )
    ax_months.set_xticklabels(list(month_ranges.keys()), fontsize=12)
    ax_months.xaxis.set_ticks_position("bottom")
    ax_months.spines["bottom"].set_position(("outward", 0))

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved figure to {output_filename}")
    plt.show()


def plot_ensemble_score_density_points(ensemble_score, test_data, title="Ensemble Score Densities", output_filename=None):
    """Plot densities of ensemble scores at specific points (0, 0.25, 0.5, 0.75, 1) in the same figure."""
    # Define the month ranges for a non-leap year
    month_ranges = {
        "Jan": (1, 31),
        "Feb": (32, 59),
        "Mar": (60, 90),
        "Apr": (91, 120),
        "May": (121, 151),
        "Jun": (152, 181),
        "Jul": (182, 212),
        "Aug": (213, 243),
        "Sep": (244, 273),
        "Oct": (274, 304),
        "Nov": (305, 334),
        "Dec": (335, 365)
    }

    # Create a DataFrame for plotting
    score_dist_data = pd.DataFrame({
        "ensemble_score": ensemble_score,
        "day": test_data["day"]
    })

    # Main figure and density plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Define specific scores to calculate densities
    score_levels = [0, 0.25, 0.5, 0.75, 1]

    # Plot densities for specific ensemble score levels
    for score in score_levels:
        sns.kdeplot(
            data=score_dist_data[score_dist_data["ensemble_score"] == score],
            x="day",
            fill=False,
            common_norm=False,
            alpha=0.7,
            label=f"Score = {score}",
            ax=ax
        )

    # Set labels and limits
    ax.set_xlabel("")
    ax.set_xticks([])  # Remove x-ticks for similar style
    ax.set_ylabel("", fontsize=12)
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 0.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Ensemble score", fontsize=10, title_fontsize=10)

    # Create a secondary axis below for the month ranges
    ax_months = ax.twiny()
    ax_months.set_xlim(ax.get_xlim())
    ax_months.set_xticks(
        [(start + end) // 2 for start, end in month_ranges.values()]
    )
    ax_months.set_xticklabels(list(month_ranges.keys()), fontsize=12)
    ax_months.xaxis.set_ticks_position("bottom")
    ax_months.spines["bottom"].set_position(("outward", 0))

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved figure to {output_filename}")
    plt.show()


def collect_evaluation_results(output_csv="./output/evaluation_results_summary.csv"):
    """Collect data from JSON files in evaluation_results and organize them into a table."""
    results_folder = "./evaluation_results"
    organized_data = []

    # Iterate through all JSON files in the folder
    for filename in os.listdir(results_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(results_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract metrics from the JSON file
            accuracy = data.get("Accuracy")
            mcc = data.get("MCC")
            best_threshold = data.get("Best Threshold")
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
                "Best Threshold": best_threshold,
                "Actual 0 Predicted 0": actual_0_pred_0,
                "Actual 0 Predicted 1": actual_0_pred_1,
                "Actual 1 Predicted 0": actual_1_pred_0,
                "Actual 1 Predicted 1": actual_1_pred_1
            })

    # Create a DataFrame and save it as a CSV
    df = pd.DataFrame(organized_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved organized evaluation results to {output_csv}")
