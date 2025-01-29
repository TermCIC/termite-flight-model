import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib as mpl
import matplotlib.font_manager as fm
import random
import os


def plot_taiwan_map(cf_file, cg_file, taiwan_shapefile):
    """
    Plots a Taiwan map with GPS locations from CF and CG Excel files.

    Parameters:
        cf_file (str): Path to the CF.xlsx file.
        cg_file (str): Path to the CG.xlsx file.
        taiwan_shapefile (str): Path to the Taiwan shapefile (GeoJSON or SHP format).
    """
    # Set the font to Arial for the entire plot
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 28  # Set default font size for all text
    mpl.rcParams['axes.titlesize'] = 24  # Title font size
    mpl.rcParams['axes.labelsize'] = 24  # Axis label font size
    mpl.rcParams['xtick.labelsize'] = 24  # X-axis tick label font size
    mpl.rcParams['ytick.labelsize'] = 24  # Y-axis tick label font size
    mpl.rcParams['legend.fontsize'] = 24  # Legend font size

    # Read the shapefile for Taiwan
    taiwan_map = gpd.read_file(taiwan_shapefile)

    # Read the GPS data from CF and CG Excel files
    cf_data = pd.read_excel(cf_file)
    cg_data = pd.read_excel(cg_file)

    # Ensure latitude and longitude columns exist
    if 'LATDD' not in cf_data.columns or 'LONDD' not in cf_data.columns:
        raise ValueError("CF file must contain 'LATDD' and 'LONDD' columns.")
    if 'LATDD' not in cg_data.columns or 'LONDD' not in cg_data.columns:
        raise ValueError("CG file must contain 'LATDD' and 'LONDD' columns.")

    # Create GeoDataFrames for CF and CG data
    cf_gdf = gpd.GeoDataFrame(cf_data, geometry=gpd.points_from_xy(cf_data['LONDD'], cf_data['LATDD']))
    cg_gdf = gpd.GeoDataFrame(cg_data, geometry=gpd.points_from_xy(cg_data['LONDD'], cg_data['LATDD']))

    # Plot the Taiwan map
    fig, ax = plt.subplots(figsize=(9, 10))
    taiwan_map.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot CF and CG GPS locations
    cf_gdf.plot(ax=ax, color='blue', markersize=10, label='Coptotermes formosanus')
    cg_gdf.plot(ax=ax, color='red', markersize=10, label='Coptotermes gestroi')

    # Create italic font properties for the legend
    italic_font = fm.FontProperties(style='italic')

    # Customize x and y axis range if specified
    ax.set_xlim((118, 122.5))
    ax.set_ylim((21.5, 26))

    # Add the legend with custom font properties
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontproperties(italic_font)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=1)

    # Add map details
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save and show the plot
    plt.savefig("./output/map.png", dpi=300)
    plt.show()

# Example usage
# Replace with actual file paths for the CF and CG Excel files and the Taiwan shapefile
cf_excel_path = "CF.xlsx"
cg_excel_path = "CG.xlsx"
taiwan_shapefile_path = "world-administrative-boundaries.geojson"  # Use a GeoJSON or SHP file for Taiwan's boundaries
plot_taiwan_map(cf_excel_path, cg_excel_path, taiwan_shapefile_path)

def generate_random_points(region_bounds, n_points):
    """
    Generate random GPS points within a specified region.

    Parameters:
        region_bounds (dict): A dictionary with 'lat_min', 'lat_max', 'lon_min', and 'lon_max'.
        n_points (int): Number of points to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the randomly generated points with 'LATDD' and 'LONDD'.
    """
    random_points = []
    for _ in range(n_points):
        lat = random.uniform(region_bounds['lat_min'], region_bounds['lat_max'])
        lon = random.uniform(region_bounds['lon_min'], region_bounds['lon_max'])
        random_points.append({'LATDD': lat, 'LONDD': lon})
    return pd.DataFrame(random_points)

def plot_random_points_on_map(taiwan_shapefile):
    """
    Generate random GPS points in north, middle-west, and south Taiwan, plot them on the map, 
    and save the points to a CSV file.

    Parameters:
        taiwan_shapefile (str): Path to the Taiwan shapefile (GeoJSON or SHP format).
    """
    # Set the font to Arial for the entire plot
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 28  # Set default font size for all text
    mpl.rcParams['legend.fontsize'] = 24  # Legend font size

    # Read the shapefile for Taiwan
    taiwan_map = gpd.read_file(taiwan_shapefile)

    # Define region bounds for north, middle-west, and south Taiwan
    region_bounds = {
        'north': {'lat_min': 24.8, 'lat_max': 25.0, 'lon_min': 121.5, 'lon_max': 121.7},
        'middle_west': {'lat_min': 24.1, 'lat_max': 24.3, 'lon_min': 120.6, 'lon_max': 120.8},
        'south': {'lat_min': 22.6, 'lat_max': 22.8, 'lon_min': 120.3, 'lon_max': 120.5},
    }

    # Generate random points for each region
    north_points = generate_random_points(region_bounds['north'], 30)
    middle_west_points = generate_random_points(region_bounds['middle_west'], 30)
    south_points = generate_random_points(region_bounds['south'], 30)

    # Combine all points into a single DataFrame
    all_points = pd.concat([north_points, middle_west_points, south_points], ignore_index=True)

    # Save the points of each region separately
    os.makedirs("./db", exist_ok=True)
    north_points.to_csv("./db/north_points.csv", index=False)
    middle_west_points.to_csv("./db/middle_west_points.csv", index=False)
    south_points.to_csv("./db/south_points.csv", index=False)

    # Create a GeoDataFrame for the random points
    random_gdf = gpd.GeoDataFrame(
        all_points, geometry=gpd.points_from_xy(all_points['LONDD'], all_points['LATDD'])
    )

    # Plot the Taiwan map
    fig, ax = plt.subplots(figsize=(9, 10))
    taiwan_map.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot random points
    random_gdf.plot(ax=ax, color='green', markersize=20, label='Randomly sampled points')

    # Customize x and y axis range if specified
    ax.set_xlim((118, 122.5))
    ax.set_ylim((21.5, 26))

    # Add legend and details
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the map with random points
    plt.savefig("./output/map_with_random_points.png", dpi=300)
    plt.show()

# Example usage
taiwan_shapefile_path = "world-administrative-boundaries.geojson"  # Use a GeoJSON or SHP file for Taiwan's boundaries
plot_random_points_on_map(taiwan_shapefile_path)