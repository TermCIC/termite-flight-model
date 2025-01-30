import os
import pandas as pd
import sqlite3
from datetime import datetime

# Define the directory containing CSV files
input_dir = "climate_analysis/AR6_TaiESM1_dataset/"

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

if not csv_files:
    print("No CSV files found in the directory.")
else:
    print(f"Found {len(csv_files)} CSV files. Processing...")

# Process each CSV file and create a corresponding SQLite database
for csv_file in csv_files:
    file_path = os.path.join(input_dir, csv_file)
    
    # Create the output database file with the same name but with .db extension
    db_filename = os.path.splitext(csv_file)[0] + ".db"
    output_db = os.path.join(input_dir, db_filename)
    
    print(f"Creating SQLite database for {csv_file} -> {db_filename} ...")
    # Create a new SQLite database
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Create table structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            longitude REAL,
            latitude REAL,
            date TEXT,
            precipitation REAL,
            temperature_mean REAL,
            temperature_max REAL,
            temperature_min REAL,
            day INTEGER,
            elevation REAL
        )
    ''')
    conn.commit()
    print(f"Database {db_filename} and table created successfully.")

    print(f"Processing {csv_file}...")
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=1000):
        print(f"  Processing chunk of size {len(chunk)}...")

        # Rename the headers
        chunk.columns = ["longitude", "latitude", "date", "precipitation", "temperature_mean", "temperature_max", "temperature_min"]

        # Convert date to datetime format and add 'day' column
        chunk['date'] = pd.to_datetime(chunk['date'], format='%Y%m%d').astype(str)
        chunk['day'] = pd.to_datetime(chunk['date']).dt.dayofyear

        # Insert data into SQLite database
        chunk.to_sql("weather_data", conn, if_exists="append", index=False)
        print(f"  Inserted {len(chunk)} rows into {db_filename}.")

    conn.commit()
    conn.close()
    print(f"Finished processing {csv_file}. Database saved as {db_filename}.\n")

print("All CSV files have been processed into individual SQLite databases.")
