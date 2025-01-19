from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import requests_cache
import time
import math
import openmeteo_requests
from retry_requests import retry
import os


def parse_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        parts = date_str.split("-")
        if len(parts) == 3:
            return pd.to_datetime(f"{parts[0]}-{parts[1]}-{parts[2]}")
        else:
            raise ValueError(f"Unrecognized date format: {date_str}")


def append_accumulation(db_path, output_db_path):
    # Connect to the original SQLite database
    conn = sqlite3.connect(db_path)

    # Read the existing data from the database into a DataFrame
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)

    # Convert the date column to datetime using the custom parsing function
    df['date'] = df['date'].apply(parse_date)

    # Sort the DataFrame by colony and date to ensure correct cumulative calculation
    df = df.sort_values(by=['colony', 'date'])

    # Calculate the day of the year for each date
    df['day'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month

    # Calculate cumulative values for each day within each colony
    df['cumulative_temperature_2m_mean'] = df.groupby(
        'colony')['temperature_2m_mean'].cumsum()
    df['cumulative_apparent_temperature_mean'] = df.groupby(
        'colony')['apparent_temperature_mean'].cumsum()
    df['cumulative_daylight_duration'] = df.groupby(
        'colony')['daylight_duration'].cumsum()
    df['cumulative_sunshine_duration'] = df.groupby(
        'colony')['sunshine_duration'].cumsum()
    df['cumulative_precipitation_sum'] = df.groupby(
        'colony')['precipitation_sum'].cumsum()
    df['cumulative_rain_sum'] = df.groupby('colony')['rain_sum'].cumsum()
    df['cumulative_precipitation_hours'] = df.groupby(
        'colony')['precipitation_hours'].cumsum()
    df['cumulative_shortwave_radiation_sum'] = df.groupby(
        'colony')['shortwave_radiation_sum'].cumsum()
    df['cumulative_et0_fao_evapotranspiration'] = df.groupby(
        'colony')['et0_fao_evapotranspiration'].cumsum()

    # Connect to the new SQLite database
    output_conn = sqlite3.connect(output_db_path)

    # Write the updated DataFrame to the new database
    df.to_sql('weather_data', output_conn, if_exists='replace', index=False)

    # Close the database connections
    conn.close()
    output_conn.close()

    print("Cumulative values and day variable calculated and saved to the new database successfully.")


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"


def fetch_climate_data(latitude, longitude, year, month, day, one_single_day=False):
    start_date = f"{year}-01-01"
    date = datetime(year, month, day)
    formatted_date = date.strftime("%Y-%m-%d")
    end_date = f"{year}-12-31"
    if one_single_day:
        start_date = formatted_date
        end_date = formatted_date

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max",
                  "apparent_temperature_min", "apparent_temperature_mean", "sunrise", "sunset", "daylight_duration",
                  "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours", "wind_speed_10m_max",
                  "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
                  "et0_fao_evapotranspiration"],
        "timezone": "Asia/Singapore"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    elevation = response.Elevation()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
    daily_apparent_temperature_min = daily.Variables(4).ValuesAsNumpy()
    daily_apparent_temperature_mean = daily.Variables(5).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(8).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(9).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(10).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(11).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(12).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(13).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(14).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(15).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(16).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(17).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}

    daily_data["elevation"] = elevation
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
    daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
    daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
    daily_data["daylight_duration"] = daily_daylight_duration
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
    return daily_data


def fetch_data_for_occurrence_day(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Define the columns to check/create
    columns_to_create = {
        "elevation": pd.NA,
        "temperature_2m_max": pd.NA,
        "temperature_2m_min": pd.NA,
        "temperature_2m_mean": pd.NA,
        "apparent_temperature_max": pd.NA,
        "apparent_temperature_min": pd.NA,
        "apparent_temperature_mean": pd.NA,
        "daylight_duration": pd.NA,
        "sunshine_duration": pd.NA,
        "precipitation_sum": pd.NA,
        "rain_sum": pd.NA,
        "precipitation_hours": pd.NA,
        "wind_speed_10m_max": pd.NA,
        "wind_gusts_10m_max": pd.NA,
        "wind_direction_10m_dominant": pd.NA,
        "shortwave_radiation_sum": pd.NA,
        "et0_fao_evapotranspiration": pd.NA
    }

    # Check if each column exists, and if not, create it
    for column, default_value in columns_to_create.items():
        if column not in df.columns:
            df[column] = default_value

    # Select column and get data
    latitudes = df["LATDD"]
    longitudes = df["LONDD"]
    dates = df["Col. Date"]

    for d in range(0, len(dates)):
        if isinstance(dates[d], datetime):
            year = dates[d].year
            month = dates[d].month
            day = dates[d].day
            latitude = latitudes[d]
            longitude = longitudes[d]
            if not pd.isna(latitude) and not pd.isna(longitude):
                print(
                    f"""fetching climate data for sample {d}: year={year}, month={month}, day={day}, LAT={latitude}, LON={longitude}""")

                data_fetched = fetch_climate_data(
                    latitude, longitude, year, month, day)
                df.at[d, "elevation"] = data_fetched['elevation']
                df.at[d, "temperature_2m_max"] = data_fetched['temperature_2m_max'][0]
                df.at[d, "temperature_2m_min"] = data_fetched['temperature_2m_min'][0]
                df.at[d, "temperature_2m_mean"] = data_fetched['temperature_2m_mean'][0]
                df.at[d, "apparent_temperature_max"] = data_fetched['apparent_temperature_max'][0]
                df.at[d, "apparent_temperature_min"] = data_fetched['apparent_temperature_min'][0]
                df.at[d, "apparent_temperature_mean"] = data_fetched['apparent_temperature_mean'][0]
                df.at[d, "daylight_duration"] = data_fetched['daylight_duration'][0]
                df.at[d, "sunshine_duration"] = data_fetched['sunshine_duration'][0]
                df.at[d, "precipitation_sum"] = data_fetched['precipitation_sum'][0]
                df.at[d, "rain_sum"] = data_fetched['rain_sum'][0]
                df.at[d, "precipitation_hours"] = data_fetched['precipitation_hours'][0]
                df.at[d, "wind_speed_10m_max"] = data_fetched['wind_speed_10m_max'][0]
                df.at[d, "wind_gusts_10m_max"] = data_fetched['wind_gusts_10m_max'][0]
                df.at[d, "wind_direction_10m_dominant"] = data_fetched['wind_direction_10m_dominant'][0]
                df.at[d, "shortwave_radiation_sum"] = data_fetched['shortwave_radiation_sum'][0]
                df.at[d, "et0_fao_evapotranspiration"] = data_fetched['et0_fao_evapotranspiration'][0]

    # Save the modified DataFrame back to an Excel file if changes were made
    df.to_excel(file_path, index=False)


# Not run if exist
def insert_occurrence_day_to_follows():
    fetch_data_for_occurrence_day("Coptotermes formosanus.xlsx")
    fetch_data_for_occurrence_day("Coptotermes gestroi.xlsx")


def load_destination_excel(destination_excel):
    # Check if destination Excel file exists, if not, create it
    if not os.path.exists(destination_excel):
        # Define the columns to create
        columns = [
            "colony", "latitude", "longitude", "date", "flight", "elevation",
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
            "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum",
            "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
            "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
        ]
        # Create an empty DataFrame with the specified columns
        destination_df = pd.DataFrame(columns=columns)
        # Save the empty DataFrame to the destination Excel file
        destination_df.to_excel(destination_excel, index=False)
    else:
        # Read the existing destination Excel file
        destination_df = pd.read_excel(destination_excel)

    return destination_df


def load_destination_sqlite(destination_db):
    conn = sqlite3.connect(destination_db)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS weather_data (
        colony TEXT,
        latitude REAL,
        longitude REAL,
        date TEXT,
        flight INTEGER,
        elevation REAL,
        temperature_2m_max REAL,
        temperature_2m_min REAL,
        temperature_2m_mean REAL,
        apparent_temperature_max REAL,
        apparent_temperature_min REAL,
        apparent_temperature_mean REAL,
        daylight_duration REAL,
        sunshine_duration REAL,
        precipitation_sum REAL,
        rain_sum REAL,
        precipitation_hours REAL,
        wind_speed_10m_max REAL,
        wind_gusts_10m_max REAL,
        wind_direction_10m_dominant REAL,
        shortwave_radiation_sum REAL,
        et0_fao_evapotranspiration REAL
    )
    ''')
    conn.commit()
    return conn


def append_whole_year_data_of_colonies(source_excel, destination_db):
    source_df = pd.read_excel(source_excel)
    conn = load_destination_sqlite(destination_db)
    cursor = conn.cursor()

    latitudes = source_df["LATDD"]
    longitudes = source_df["LONDD"]
    dates = source_df["Col. Date"]

    for d in range(0, len(dates)):
        if isinstance(dates[d], datetime):
            year = dates[d].year
            month = dates[d].month
            day = dates[d].day
            flight_date = dates[d].strftime("%Y-%m-%d")
            latitude = latitudes[d]
            longitude = longitudes[d]
            colony = f"{latitude}-{longitude}-{flight_date}"
            cursor.execute(
                'SELECT COUNT(*) FROM weather_data WHERE colony = ?', (colony,))
            if cursor.fetchone()[0] == 0:
                if not math.isnan(latitude):
                    try:
                        print(
                            f"Colony: {colony} not in db, waiting for fetching data.")
                        for i in range(5):
                            print(f"-> {i}/5 second")
                            time.sleep(1)
                        if not pd.isna(latitude) and not pd.isna(longitude):
                            if year not in [2024, "2024"]:
                                print(
                                    f"Fetching climate data for sample {d}: year={year}, LAT={latitude}, LON={longitude}")
                                data_fetched = fetch_climate_data(
                                    latitude, longitude, year, month, day)
                                for i in range(len(data_fetched['date'])):
                                    date = data_fetched['date'][i].strftime(
                                        "%Y-%m-%d")
                                    flight = 1 if date == flight_date else 0
                                    row = (
                                        colony,
                                        float(latitude),
                                        float(longitude),
                                        date,
                                        flight,
                                        float(data_fetched['elevation']),
                                        float(
                                            data_fetched['temperature_2m_max'][i]),
                                        float(
                                            data_fetched['temperature_2m_min'][i]),
                                        float(
                                            data_fetched['temperature_2m_mean'][i]),
                                        float(
                                            data_fetched['apparent_temperature_max'][i]),
                                        float(
                                            data_fetched['apparent_temperature_min'][i]),
                                        float(
                                            data_fetched['apparent_temperature_mean'][i]),
                                        float(
                                            data_fetched['daylight_duration'][i]),
                                        float(
                                            data_fetched['sunshine_duration'][i]),
                                        float(
                                            data_fetched['precipitation_sum'][i]),
                                        float(data_fetched['rain_sum'][i]),
                                        float(
                                            data_fetched['precipitation_hours'][i]),
                                        float(
                                            data_fetched['wind_speed_10m_max'][i]),
                                        float(
                                            data_fetched['wind_gusts_10m_max'][i]),
                                        float(
                                            data_fetched['wind_direction_10m_dominant'][i]),
                                        float(
                                            data_fetched['shortwave_radiation_sum'][i]),
                                        float(
                                            data_fetched['et0_fao_evapotranspiration'][i])
                                    )
                                    cursor.execute('''
                                    INSERT INTO weather_data (
                                        colony, latitude, longitude, date, flight, elevation,
                                        temperature_2m_max, temperature_2m_min, temperature_2m_mean,
                                        apparent_temperature_max, apparent_temperature_min, apparent_temperature_mean,
                                        daylight_duration, sunshine_duration, precipitation_sum, rain_sum,
                                        precipitation_hours, wind_speed_10m_max, wind_gusts_10m_max,
                                        wind_direction_10m_dominant, shortwave_radiation_sum, et0_fao_evapotranspiration
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', row)
                                    conn.commit()
                    except Exception as e:
                        print(
                            f"Seems API not working now, try again later...({e})")
                        for i in range(300):
                            print(f"-> {i}/300 second")
                            time.sleep(1)
            else:
                print("Colony exists!")
        else:
            print("Date format wrong!")

    conn.close()

# Examples
# append_whole_year_data_of_colonies(source_excel="CF.xlsx", destination_db="CF.db")
# append_whole_year_data_of_colonies(source_excel="Test2.xlsx", destination_db="Test2.db")
# append_accumulation("Test2.db", "Test2.db")
# append_whole_year_data_of_colonies(source_excel="CG.xlsx", destination_db="CG.db")
# append_accumulation("CG.db", "CG_cumulative.db")
