import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------
# 1. SET PLOT STYLES
# -------------------------------
sns.set_style("whitegrid")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = False  # Set to True if you want LaTeX rendering
mpl.rc("xtick", labelsize=10)
mpl.rc("ytick", labelsize=10)
plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"legend.fontsize": 10})

# -------------------------------
# 2. LOAD THE INDIA SHAPEFILE
# -------------------------------
shapefile_path = "culture-evaluation/our_csis/india_map/india-polygon.shp"  # Adjust path if needed
map_df = gpd.read_file(shapefile_path)

# -------------------------------
# 3. DEFINE THE CUSTOM COLORMAP (WHITE -> VIOLET)
# -------------------------------
violet_white = LinearSegmentedColormap.from_list("violet_white", ["white", "#8A2BE2"], N=256)

# -------------------------------
# 4. FUNCTION TO MERGE CSV DATA WITH SHAPEFILE
# -------------------------------
def create_heatmap_data(map_gdf, data_df, model_col):
    """
    Merges the shapefile (using 'st_nm' as state name) with a CSV column.
    Expects the CSV to have a "States" column.
    """
    temp = data_df[["States", model_col]].copy()
    temp = temp.rename(columns={"States": "State", model_col: "Score"})
    merged = map_gdf.set_index("st_nm").join(temp.set_index("State"))
    merged["Score"] = merged["Score"].fillna(0)
    return merged

# -------------------------------
# 5. SET DIRECTORIES FOR CSV INPUT AND PDF OUTPUT
# -------------------------------
csv_dir = "culture-evaluation/our_csis/csis"
# Get all CSV files that end with _adaptation_scores.csv from csv_dir
csv_files = glob.glob(os.path.join(csv_dir, "*_adaptation_scores.csv"))

# Choose output directory for PDFs (here we use the india_map folder)
output_dir = "culture-evaluation/our_csis/india_map"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------------
# 6. PROCESS EACH CSV FILE AND SAVE HEATMAPS
# -------------------------------
for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    # Load the CSV data
    df = pd.read_csv(csv_file)
    # Remove the "Overall" row to avoid merging issues
    df = df[df["States"] != "Overall"].copy()
    # Identify model columns (everything except "States")
    model_cols = [col for col in df.columns if col != "States"]
    
    # Set the color range to be 0-1
    vmin_val = 0.0
    vmax_val = 1.0
    
    # Determine grid dimensions for subplots.
    n = len(model_cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    
    # Create subplots with increased figure size: 20 x 20
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 9))
    axs = axs.flatten()
    
    # Plot each model's data using the violet-white colormap.
    for i, model in enumerate(model_cols):
        if i >= len(axs):
            break  # In case there are more models than axes
        ax = axs[i]
        merged_gdf = create_heatmap_data(map_df, df, model)
        merged_gdf.plot(
            column="Score",
            cmap=violet_white,
            linewidth=0.8,
            edgecolor="black",
            vmin=vmin_val,
            vmax=vmax_val,
            missing_kwds={"color": "lightgray", "label": "No Data"},
            ax=ax
        )
        ax.axis("off")
        ax.set_title(model, fontsize=15)
    
    # Hide any extra subplots if there are unused axes.
    for j in range(len(model_cols), len(axs)):
        axs[j].axis("off")
    
    # Add a single common horizontal colorbar at the bottom.
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    norm = mpl.colors.Normalize(vmin=vmin_val, vmax=vmax_val)
    sm = plt.cm.ScalarMappable(cmap=violet_white, norm=norm)
    sm.set_array([])
    # Create an axis for the colorbar in figure coordinates.
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Adaptation Score", labelpad=5)
    
    # Construct output filename: remove "_adaptation_scores" from the CSV basename.
    base_name = os.path.splitext(os.path.basename(csv_file))[0].replace("_adaptation_scores", "")
    output_pdf = os.path.join(output_dir, f"{base_name}_heatmap.pdf")
    
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved heatmap as {output_pdf}")
