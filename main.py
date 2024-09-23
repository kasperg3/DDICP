import json

import contextily as cx
import matplotlib.pyplot as plt
import shapely
import trajgenpy.Logging
from trajgenpy import Utils
from trajgenpy.Geometries import (
    GeoMultiPolygon,
    GeoMultiTrajectory,
    GeoPolygon
)
import cv2
from trajgenpy.Query import query_features
import numpy as np
from scipy.ndimage import gaussian_filter, fourier_gaussian
from shapely.geometry import LineString
from shapely.ops import transform
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
from PIL import Image
import rasterio.features
log = trajgenpy.Logging.get_logger()
# Function to interpolate points on the line
def interpolate_line(line, distance):
    num_points = int(line.length / distance)
    points = []
    if num_points != 0:
        points = [line.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
    else: 
        points = [line.interpolate(2, normalized=True)]
    return points

# # Apply the heatmap generation to all road lines
def generate_heatmap(geometry_collection, sample_distance, xedges, yedges, exterior_feature = True):
    heatmap = np.zeros((len(xedges) - 1, len(yedges) - 1))
    for feature in geometry_collection.geoms:
        interpolated_points = []
        if isinstance(feature, LineString):
            line = feature
            interpolated_points = interpolate_line(line, sample_distance)
        elif isinstance(feature, shapely.geometry.Polygon):        
            if exterior_feature:
                line = feature.exterior
                interpolated_points = interpolate_line(line, sample_distance)                
            else:     
                # Check if all combinations of xedges and yedges are inside the polygon
                x_y_combinations = np.array(np.meshgrid(xedges, yedges)).T.reshape(-1, 2)
                for x, y in x_y_combinations:
                    point = shapely.geometry.Point(x, y)
                    if feature.contains(point):
                        interpolated_points.append(point)
            
        x, y = zip(*[(point.x, point.y) for point in interpolated_points])
        
        temp_heatmap, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap += temp_heatmap
    return heatmap

def normalize_heatmap(heatmap, norm):
    # Normalize or clip the heatmap
    if norm == "clip":
        heatmap = np.clip(heatmap, 0, 1)
    elif norm== "normalize":
        heatmap = heatmap / np.max(heatmap)
    return heatmap

def interactive_plot(polygon_file,norm, use_sliders=True, plot_environment = True, export = False):
    # Load the GeoJSON data
    with open(polygon_file, "r") as f:
        data = json.load(f)

    # Extract the polygon coordinates
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    query_region = shapely.Polygon(coordinates)
    # Create the shapely Polygon object
    polygon = GeoPolygon(query_region).set_crs("EPSG:2197")

    wetland = query_features(
        GeoPolygon(query_region),
        {
            "natural": ["water", "wetland"],
        },
    )
    wetland_collection = []
    for feature in wetland.values():
        wetland_collection.extend(list(feature.geoms))
    wetland_geom = GeoMultiPolygon(wetland_collection).set_crs("EPSG:2197")
    
    roads = query_features(
        GeoPolygon(query_region),
        {
            "highway": ["service", "track"],
        },
    )
    road_collection = []
    for road in roads.values():
        road_collection.extend(list(road.geoms))
    road_geom = GeoMultiTrajectory(road_collection).set_crs("EPSG:2197")
    
    building_collection = []
    buildings = query_features(
        GeoPolygon(query_region),
        {"building": True},
    )
    for feature in buildings.values():
        building_collection.extend(list(feature.geoms))
    building_geom = GeoMultiPolygon(building_collection).set_crs("EPSG:2197")
    
    # Determine the bounds of the polygon
    sample_distance = 1
    minx, miny, maxx, maxy = polygon.geometry.bounds

    # Define the number of bins for the heatmap
    num_bins = 300

    buffer = 100 # in meters
    # Create the edges for the histogram bins
    xedges = np.linspace(minx-buffer, maxx+buffer, num_bins + 1)
    yedges = np.linspace(miny-buffer, maxy+buffer, num_bins + 1)

    # Generate heatmap for road geometries
    road_heatmap= generate_heatmap(road_geom.geometry, sample_distance, xedges, yedges)
    wetland_heatmap= generate_heatmap(wetland_geom.geometry, sample_distance, xedges,yedges)
    
    heatmaps = {"roads": road_heatmap, "wetland" :wetland_heatmap}
    # Apply Gaussian filter to the combined heatmap
    heatmap = gaussian_filter(road_heatmap + wetland_heatmap, sigma=3)
    heatmap = normalize_heatmap(heatmap, norm)
    # Create a figure and axis for the slider
    fig, ax = plt.subplots()
    # Set xlim and ylim based on the bounding box of the polygon
    # to ensure the basemap has the right location
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    if plot_environment:
        # Ploting 
        road_geom.plot(color="red",
            linestyle="dashed",)
        wetland_geom.plot()
        # polygon.plot(facecolor="none", edgecolor="black", linewidth=2)
        polygon.plot(
            facecolor="none",
            edgecolor="black",
            linewidth=2,
        )
        # building_geom.plot()
        
        
        alpha = 0.4
        Utils.plot_basemap(provider=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:2197")
        # colorscheme
        # Flare
        # colors = [(1, 1, 1)] + sns.color_palette("flare", as_cmap=False)
        # Custom colors
        # colors = [
        #     (1, 1, 1),  # white
        #     # (0, 0, 1),  # blue
        #     # (0, 1, 1),  # cyan
        #     # (0, 1, 0),  # green
        #     # (1, 1, 0),  # yellow
        #     (1, 0, 0),  # red
        #     (0.5, 0, 0.5),  # purple
        #     (0, 0, 0)  # purple
        # ]
        # palette = LinearSegmentedColormap.from_list("custom_palette", colors)
        # JET
        colors =  [mcolors.to_rgb(c) for c in plt.cm.jet(np.linspace(0, 1, 256))]
        palette = LinearSegmentedColormap.from_list("custom_flare", colors)
    else: 
        alpha = 1
        palette = "gist_gray"

    # Use the custom colormap for the heatmap
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=palette), ax=ax, orientation='vertical')
    cbar.set_label('Target Distribution')
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap_img = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=palette, alpha=alpha)
    

    if use_sliders:
        filter_sliders = {}
        multiplier_sliders = {}
        for key in heatmaps.keys():
            y_position = 0+ 0.05 * list(heatmaps.keys()).index(key)
            
            ax_filter_slider = plt.axes([0.25, y_position, 0.25, 0.03])
            ax_multiplier_slider = plt.axes([0.60, y_position, 0.25, 0.03])
            
            filter_sliders[key] = Slider(ax_filter_slider, f'{key}:  σ', 0.0, 10.0, valinit=3, valstep=0.1)
            multiplier_sliders[key] = Slider(ax_multiplier_slider, 'α', 0.0, 3.0, valinit=1, valstep=0.1)

        def update(val):
            combined_heatmap = np.zeros_like(road_heatmap)
            for key, slider in filter_sliders.items():
                combined_heatmap += gaussian_filter(heatmaps[key]*multiplier_sliders[key].val, sigma=slider.val)
            combined_heatmap = normalize_heatmap(combined_heatmap, norm)
            heatmap_img.set_data(combined_heatmap.T)
            plt.draw()

        # Add a save button
        save_ax = plt.axes([0.5, 0.9, 0.1, 0.04])
        
        save_button = plt.Button(save_ax, 'Save', color='white', hovercolor='0.975')

        def save(event):
            combined_heatmap = np.zeros_like(road_heatmap)
            for key, slider in filter_sliders.items():
                combined_heatmap += gaussian_filter(heatmaps[key]*multiplier_sliders[key].val, sigma=slider.val)
            combined_heatmap = normalize_heatmap(combined_heatmap, norm)
            temp_heatmap = np.flipud(combined_heatmap.T) # Makes sure that the map is oriented correctly
            
            height, width = temp_heatmap.shape
            greyscale_with_alpha = np.zeros((height, width, 2), dtype=np.uint8)
            # Set the grayscale channel based on the matrix values
            greyscale_with_alpha[..., 0] = temp_heatmap*255  # Greyscale (intensity) values
            # Set the alpha channel: 255 where matrix > 0, otherwise 0 (transparent)
            greyscale_with_alpha[..., 1] = np.where(temp_heatmap > 0.01, 255, 0)
            
            # Convert to a greyscale image with alpha
            img = Image.fromarray(greyscale_with_alpha, mode='LA')
            img.save("heatmap.png")
            print("Heatmap saved as heatmap.png")

        save_button.on_clicked(save)
        
        for slider in filter_sliders.values():
            slider.on_changed(update)
        for slider in multiplier_sliders.values():
            slider.on_changed(update)
    
    # plt.axis("equal")
    ax.set_axis_off()

    if export:
        plt.savefig("plot.png",bbox_inches='tight')
    plt.show()
    

# TODO Look into checkboxes for enabling and disabling environmental features: https://gist.github.com/DataSolveProblems/143e2c6f5ecd2c0b4876ac4308e7a2d0

if __name__ == "__main__":
    use_sliders = True
    plot_environment = True
    norm = "clip" # Options: "normalize", "clip", None
    polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    interactive_plot(polygon_file, norm, use_sliders, plot_environment, export=True)