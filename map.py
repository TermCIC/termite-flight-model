import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib as mpl
import matplotlib.font_manager as fm

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
    mpl.rcParams['font.size'] = 16  # Set default font size for all text
    mpl.rcParams['axes.titlesize'] = 20  # Title font size
    mpl.rcParams['axes.labelsize'] = 18  # Axis label font size
    mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick label font size
    mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick label font size
    mpl.rcParams['legend.fontsize'] = 16  # Legend font size

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
    fig, ax = plt.subplots(figsize=(10, 10))
    taiwan_map.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot CF and CG GPS locations
    cf_gdf.plot(ax=ax, color='blue', markersize=10, label='Coptotermes formosanus')
    cg_gdf.plot(ax=ax, color='red', markersize=10, label='Coptotermes gestroi')

    # Create italic font properties for the legend
    italic_font = fm.FontProperties(style='italic')

    # Add the legend with custom font properties
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontproperties(italic_font)

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
