import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

    # Main figure and density plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Kernel Density Plot
    sns.kdeplot(
        data=test_data,
        x="day",
        hue=target_column,
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={0: "blue", 1: "red"},
        ax=ax
    )

    # Set labels and limits
    ax.set_xlabel("")
    # ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(1, 365)
    ax.set_xticks([])  # Remove x-ticks for similar style
    ax.set_ylim(0, 0.03)  # Allow y-axis to adjust automatically
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
