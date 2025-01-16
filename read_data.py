import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

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
required_columns = [
    "flight", "elevation", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "daylight_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "latitude", "longitude", "day", "cumulative_temperature_2m_mean",
    "cumulative_apparent_temperature_mean", "cumulative_daylight_duration",
    "cumulative_sunshine_duration", "cumulative_precipitation_sum", "cumulative_rain_sum",
    "cumulative_precipitation_hours", "cumulative_shortwave_radiation_sum",
    "cumulative_et0_fao_evapotranspiration"
]


# Function to check for missing columns and select required columns
def prepare_data(data, required_columns):
    """Check for missing columns and prepare the dataset."""
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
    return data[required_columns].dropna()


# Function to split the data into training and testing sets
def split_data(data, target_column="flight", downsample_fraction=0.01, test_size=0.3):
    """Split data into training and testing sets."""
    data[target_column] = data[target_column].astype('category')
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[target_column])
    flight_0 = train_data[train_data[target_column] == 0]
    flight_not_0 = train_data[train_data[target_column] != 0]
    sampled_flight_0 = flight_0.sample(
        frac=downsample_fraction, random_state=123)
    train_data = pd.concat([sampled_flight_0, flight_not_0], ignore_index=True)
    return train_data, test_data


# Prepare the datasets (cf_data and cg_data)
cf_data = prepare_data(cf_data, required_columns)
cg_data = prepare_data(cg_data, required_columns)

# Split the data into training and testing sets
cf_train_data, cf_test_data = split_data(cf_data)
cg_train_data, cg_test_data = split_data(cg_data)

# Display the first few rows of training data for verification
print("CF Training Data:")
print(cf_train_data.head())

print("\nCG Training Data:")
print(cg_train_data.head())
