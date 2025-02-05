from evaluate_ensemble_models import cf_ensemble, cg_ensemble
from read_data import cf_test_data, cg_test_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_flight_frequency_density(test_data, output_filename=None):
    """Generate a kernel density plot for the frequency of flight events in the dataset."""
    # Ensure the target column is numeric
    test_data["flight"] = test_data["flight"].astype(int)
    test_data["Flight"] = test_data["flight"]

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


def plot_ensemble_prediction_density(ensemble_score, test_data, output_filename=None):
    """Generate a density plot showing the distribution of ensemble predictions (0 and 1)."""
    
    # Convert ensemble scores to binary predictions
    test_data["Prediction"] = (ensemble_score >= 0.5).astype(int)

    # Main figure and density plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Kernel Density Plot
    sns.kdeplot(
        data=test_data,
        x="day",
        hue="Prediction",
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={0: "blue", 1: "red"},  # Red for predicted 1, Blue for predicted 0
        ax=ax
    )

    # Set labels and limits
    ax.set_xlabel("")
    ax.set_ylabel("", fontsize=12)
    ax.set_xlim(1, 365)
    ax.set_xticks([])  # Remove x-ticks for a similar style
    ax.set_ylim(0, 0.05)  # Adjust density range
    ax.grid(True, linestyle="--", alpha=0.6)

    # Define month labels for x-axis
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
    
    # Save the figure if output_filename is provided
    if output_filename:
        plt.savefig(output_filename, dpi=300)
        print(f"Saved figure to {output_filename}")

    plt.show()



# Plot for CF dataset
plot_flight_frequency_density(
    cf_test_data, output_filename="output/cf_flight_density.png")
plot_ensemble_prediction_density(
    cf_ensemble, cf_test_data, output_filename="output/cf_ensemble_score.png")

# Plot for CG dataset
plot_flight_frequency_density(
    cg_test_data, output_filename="output/cg_flight_density.png")
plot_ensemble_prediction_density(
    cg_ensemble, cg_test_data, output_filename="output/cg_ensemble_score.png")
