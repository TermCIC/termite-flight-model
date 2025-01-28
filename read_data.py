import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Ensure the output folder exists
os.makedirs("output", exist_ok=True)


# Define the database paths
cf_path = "./db/CF_cumulative.db"
cg_path = "./db/CG_cumulative.db"


# Fetch data from the SQLite database
def fetch_data_from_db(db_path, table_name):
    """Fetch data from an SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return data


# Fetch data from the two databases
cf_data = fetch_data_from_db(cf_path, "weather_data")
cg_data = fetch_data_from_db(cg_path, "weather_data")


# Display the first few rows of the data
print("CF Data:")
print(cf_data.head())
print("\nCG Data:")
print(cg_data.head())


# Define the list of required columns
#required_columns = [
#    "flight", "elevation", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
#    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
#    "daylight_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
#    "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum",
#    "et0_fao_evapotranspiration", "latitude", "longitude", "day", "cumulative_temperature_2m_mean",
#    "cumulative_apparent_temperature_mean", "cumulative_daylight_duration",
#    "cumulative_sunshine_duration", "cumulative_precipitation_sum", "cumulative_rain_sum",
#    "cumulative_precipitation_hours", "cumulative_shortwave_radiation_sum",
#    "cumulative_et0_fao_evapotranspiration"
#]

required_columns = [
    "flight", "elevation", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "latitude", "longitude", "day"
    ]

# Function to check for missing columns and select required columns
def prepare_data(data, required_columns):
    """Check for missing columns and prepare the dataset."""
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    return data[required_columns].dropna()


def split_data(data, target_column="flight", test_size=0.25, downsample_ratio=3/365, save_path="splitted_data"):
    # Ensure the target column is categorical for stratification
    data[target_column] = data[target_column].astype('category')

    # Split into training and testing sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[target_column], random_state=168
    )

    print(f"Original dataset size: {len(data)}")
    print(f"Training dataset size: {len(train_data)}")
    print(f"Testing dataset size: {len(test_data)}")

    # Separate majority and minority classes
    flight_0 = train_data[train_data[target_column] == 0]
    flight_not_0 = train_data[train_data[target_column] != 0]

    print(f"Training 'flight=0' size before downsampling: {len(flight_0)}")
    print(f"Training 'flight!=0' size: {len(flight_not_0)}")

    # Downsample the majority class
    sampled_flight_0 = flight_0.sample(
        frac=downsample_ratio, random_state=123
    )

    print(
        f"Training 'flight=0' size after downsampling: {len(sampled_flight_0)}")

    # Combine downsampled majority class with the minority class
    train_data = pd.concat([sampled_flight_0, flight_not_0], ignore_index=True)

    print(f"Final training dataset size after downsampling: {len(train_data)}")

    # Save datasets as CSV
    train_csv_path = f"{save_path}/train_data.csv"
    test_csv_path = f"{save_path}/test_data.csv"

    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)

    print(f"Training data saved to: {train_csv_path}")
    print(f"Testing data saved to: {test_csv_path}")

    return train_data, test_data


def save_dataset_summary(train_data, test_data, prefix, output_folder="output"):
    """Save a summary of the dataset."""
    # Convert 'flight' column to numeric for summation
    train_flight_count = train_data["flight"].astype(int).sum()
    test_flight_count = test_data["flight"].astype(int).sum()

    summary = {
        "Dataset": [prefix],
        "Total Data": [len(train_data) + len(test_data)],
        "Training Data": [len(train_data)],
        "Testing Data": [len(test_data)],
        "Flight Events in Training Data": [train_flight_count],
        "Flight Events in Testing Data": [test_flight_count]
    }
    summary_df = pd.DataFrame(summary)
    summary_csv_path = f"{output_folder}/{prefix}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved dataset summary to: {summary_csv_path}")


def generate_summary_table(data, prefix, output_folder="output"):
    """
    Generate a summary table with mean values and ranges for each variable.

    Parameters:
    - data: pd.DataFrame - The dataset to summarize.
    - prefix: str - A prefix to identify the dataset (e.g., CF or CG).
    - output_folder: str - Directory to save the summary table.

    Returns:
    - summary_df: pd.DataFrame - The summary table as a DataFrame.
    """
    # Exclude non-numeric columns from the summary
    numeric_data = data.select_dtypes(include=[np.number])

    # Compute the mean, minimum, and maximum for each numeric column
    summary = {
        "Variable": numeric_data.columns,
        "Mean": numeric_data.mean().values,
        "Min": numeric_data.min().values,
        "Max": numeric_data.max().values
    }

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary)

    # Save the summary table as a CSV
    summary_csv_path = f"{output_folder}/{prefix}_variable_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved variable summary to: {summary_csv_path}")

    return summary_df


# Prepare the datasets (cf_data and cg_data)
cf_data = prepare_data(cf_data, required_columns)
cg_data = prepare_data(cg_data, required_columns)

# Split the data into training and testing sets
cf_train_data, cf_test_data = split_data(cf_data)
cg_train_data, cg_test_data = split_data(cg_data)

# Save dataset summaries
save_dataset_summary(cf_train_data, cf_test_data, prefix="CF")
save_dataset_summary(cg_train_data, cg_test_data, prefix="CG")

# Generate summaries for CF and CG datasets
cf_summary = generate_summary_table(cf_data, prefix="CF")
cg_summary = generate_summary_table(cg_data, prefix="CG")

# Display the first few rows of training data for verification
print("CF Training Data:")
print(cf_train_data.head())

print("\nCG Training Data:")
print(cg_train_data.head())
