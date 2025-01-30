import os
import sqlite3
import rasterio
import numpy as np
import pandas as pd

# Define the directory containing SQLite database files
db_dir = "climate_analysis/AR6_TaiESM1_dataset/"
elevation_tif = "wc2.1_30s_elev.tif"

# Load the elevation raster once
print("Loading elevation raster into memory...")
dataset = rasterio.open(elevation_tif)
elevation_data = dataset.read(1)  # Load elevation values
print("Elevation raster loaded successfully!")

def get_elevation(lat, lon):
    """
    Fetches elevation data from the preloaded raster for given latitude and longitude.
    """
    try:
        # Convert lat/lon to row/col indices in the raster
        row, col = dataset.index(lon, lat)  # Raster uses (longitude, latitude)

        # Check if indices are within the raster bounds
        if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
            print(f"  Out of bounds: LAT={lat}, LON={lon}. Returning NULL.")
            return None  # Return NULL instead of 0 to signify missing data

        elevation = elevation_data[row, col]  # Fetch elevation value

        # Check if elevation is valid
        if np.isnan(elevation):
            print(f"  No valid elevation found for LAT={lat}, LON={lon}. Returning NULL.")
            return None
        
        return float(elevation)  # Ensure correct type is returned
    except Exception as e:
        print(f"  Error processing LAT={lat}, LON={lon}: {e}")
        return None  # Return NULL in case of an error

def update_missing_elevations(db_dir):
    """
    Reads each SQLite database in the given directory and updates missing elevation values.
    """
    print("Scanning directory for SQLite databases...")
    db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")]
    
    if not db_files:
        print("No SQLite databases found in the directory. Exiting.")
        return
    
    print(f"Found {len(db_files)} databases. Starting elevation updates...")

    for db_file in db_files:
        db_path = os.path.join(db_dir, db_file)
        print(f"\nProcessing {db_file}...")

        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch entries with missing elevation values
        print("  Fetching records with missing elevation...")
        query = "SELECT id, latitude, longitude FROM weather_data WHERE elevation IS NULL OR elevation = 0"
        missing_entries = pd.read_sql_query(query, conn)

        if missing_entries.empty:
            print(f"  No missing elevation values in {db_file}. Skipping.")
            conn.close()
            continue
        
        print(f"  Found {len(missing_entries)} records with missing elevation.")

        # Process missing elevations in bulk
        elevation_updates = []
        
        for _, row in missing_entries.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            elevation = get_elevation(lat, lon)
            if elevation is not None:
                elevation_updates.append((elevation, row["id"]))
                print(f"  Updating ID={row['id']} -> Elevation={elevation}")

        # Bulk update database with fetched elevation values
        if elevation_updates:
            print("  Updating database with new elevation values...")
            cursor.executemany("UPDATE weather_data SET elevation = ? WHERE id = ?", elevation_updates)
            conn.commit()

        conn.close()
        print(f"  Finished updating elevations in {db_file}.")

    print("\nAll databases have been processed. Elevation updates completed.")

# Run the function to update missing elevations
update_missing_elevations(db_dir)

# Close raster dataset after processing
dataset.close()
print("Elevation raster closed. Processing complete.")
