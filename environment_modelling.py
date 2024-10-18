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
from trajgenpy.Query import query_features
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel
log = trajgenpy.Logging.get_logger()


class EnvironmentBuilder:
    def __init__(self):
        self.polygon_file = None
        self.sample_distance = 1
        self.meter_per_bin = 3
        self.buffer = 0
        self.tags = {}

    def set_polygon_file(self, polygon_file):
        self.polygon_file = polygon_file
        return self

    def set_sample_distance(self, sample_distance):
        self.sample_distance = sample_distance
        return self

    def set_meter_per_bin(self, meter_per_bin):
        self.meter_per_bin = meter_per_bin
        return self

    def set_buffer(self, buffer):
        self.buffer = buffer
        return self

    def set_feature(self, name, tags):
        self.tags[name] = tags
        return self
    
    def build(self):
        return Environment(
            self.polygon_file,
            self.sample_distance,
            self.meter_per_bin,
            self.buffer,
            self.tags
        )


class Environment:
    def __init__(self, polygon_file, sample_distance, meter_per_bin, buffer, tags):
        self.polygon_file = polygon_file
        self.sample_distance = sample_distance # The distance in pixels between each sample on an interpolated line
        self.meter_per_bin = meter_per_bin
        self.buffer = buffer
        self.polygon = None
        self.xedges = None
        self.yedges = None
        self.heatmaps = {}
        
        with open(self.polygon_file, "r") as f:
            data = json.load(f)
        coordinates = data["features"][0]["geometry"]["coordinates"][0]
        query_region = shapely.Polygon(coordinates)
        log.info("Query region bounds: %s", query_region.bounds)
        self.polygon = GeoPolygon(query_region).set_crs("EPSG:2197")
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds
        num_bins_x = int((self.maxx - self.minx) * 1 / self.meter_per_bin)
        num_bins_y = int((self.maxy - self.miny) * 1 / self.meter_per_bin)
        log.info("Number of bins x: %i y: %i", num_bins_x, num_bins_y)
        self.xedges = np.linspace(self.minx - self.buffer, self.maxx + self.buffer, num_bins_x + 1)
        self.yedges = np.linspace(self.miny - self.buffer, self.maxy + self.buffer, num_bins_y + 1)
        # 
        for key in tags.keys():
            features = query_features(
                GeoPolygon(query_region),
                tags[key],
            )
            
            feature_collection = [geom for feature in features.values() for geom in feature.geoms]
            if all(isinstance(geom, shapely.geometry.Polygon) for geom in feature_collection):
                feature_geom = GeoMultiPolygon(feature_collection).set_crs("EPSG:2197")
            elif all(isinstance(geom, LineString) for geom in feature_collection):
                feature_geom = GeoMultiTrajectory(feature_collection).set_crs("EPSG:2197")
            else:
                raise ValueError("Invalid feature type in feature collection")
            self.heatmaps[key] = self.generate_heatmap(feature_geom.geometry, self.sample_distance)
            

    def image_to_world(self, x, y):
        x = x* self.meter_per_bin + self.minx - self.buffer
        y = y* self.meter_per_bin + self.miny - self.buffer
        return x, y

    # Function to interpolate points on the line
    def interpolate_line(self, line, distance):
        num_points = int(line.length / distance)
        points = []
        if num_points != 0:
            points = [line.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
        else: 
            points = [line.interpolate(2, normalized=True)]
        return points

    # # Apply the heatmap generation to all road lines
    def generate_heatmap(self, geometry_collection, sample_distance, infill_geometries = True):
        heatmap = np.zeros((len(self.xedges) - 1, len(self.yedges) - 1))
        for feature in geometry_collection.geoms:
            interpolated_points = []
            if isinstance(feature, LineString):
                line = feature
                interpolated_points = self.interpolate_line(line, sample_distance)
            elif isinstance(feature, shapely.geometry.Polygon):        
                if infill_geometries:
                    # Check if all combinations of xedges and yedges are inside the polygon
                    x_y_combinations = np.array(np.meshgrid(self.xedges, self.yedges)).T.reshape(-1, 2)
                    mask = np.array([feature.contains(shapely.geometry.Point(x, y)) for x, y in x_y_combinations])
                    interpolated_points.extend([shapely.geometry.Point(x, y) for (x, y), m in zip(x_y_combinations, mask) if m])
                else:
                    line = feature.exterior
                    interpolated_points = self.interpolate_line(line, sample_distance)
                
            x, y = zip(*[(point.x, point.y) for point in interpolated_points])
            
            temp_heatmap, _, _ = np.histogram2d(x, y, bins=(self.xedges, self.yedges))
            heatmap += temp_heatmap
            
        # Make sure that overlapping features are not counted multiple times in the histogram
        heatmap = np.clip(heatmap, 0, 1) 
        return heatmap

    

    def get_combined_heatmap(self, sigma_features, alpha_features):
        heatmap = np.zeros((len(self.xedges) - 1, len(self.yedges) - 1))
            
        for key in self.heatmaps.keys():
            # normalize the histograms with the number of bins occupied
            heatmap += gaussian_filter((self.heatmaps[key]* alpha_features[key]) / np.sum(self.heatmaps[key]), sigma=sigma_features[key]) 
            # apply a RBF kernel to the heatmap
            # rbf_kernel = np.exp(-0.5 * (self.heatmaps[key] / np.max(self.heatmaps[key]))**2 / sigma_features[key]**2)
            # heatmap += rbf_kernel * alpha_features[key]
        
        # Make sure the probabilities sum to 1
        heatmap = heatmap / np.sum(heatmap)
        
        return heatmap
    
    def interactive_plot(self, use_sliders=True, plot_environment=True, export=False):
        # Apply Gaussian filter to the combined heatmap
        # Calculate the width of the Gaussian filter in pixels
        filter_width_meters = 10  # Example width in meters
        filter_width_pixels = filter_width_meters / self.meter_per_bin
        
        # Calculate the sigma value for the Gaussian filter
        sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
        
        sigma_features = {key: sigma for key in self.heatmaps.keys()}
        alpha_features = {key: 1 for key in self.heatmaps.keys()}
        
        heatmap = self.get_combined_heatmap(sigma_features, alpha_features)
        
        # Create a figure and axis for the slider
        fig, ax = plt.subplots()
        # Set xlim and ylim based on the bounding box of the polygon
        # to ensure the basemap has the right location
        ax.set_xlim(self.minx - self.buffer, self.maxx + self.buffer)
        ax.set_ylim(self.miny - self.buffer, self.maxy + self.buffer)
        if plot_environment:
            # Ploting 
            # road_geom.plot(color="red",linestyle="dashed",)
            # wetland_geom.plot()
            # polygon.plot(facecolor="none", edgecolor="black", linewidth=2)
            self.polygon.plot(
                facecolor="none",
                edgecolor="black",
                linewidth=2,
            )
            # building_geom.plot()
            
            
            alpha = 0.4
            Utils.plot_basemap(provider=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:2197")
            colors =  [mcolors.to_rgb(c) for c in plt.cm.jet(np.linspace(0, 1, 256))]
            palette = LinearSegmentedColormap.from_list("custom_flare", colors)
            # palette = plt.cm.coolwarm
        else: 
            alpha = 1
            palette = "gist_gray"

        # Use the custom colormap for the heatmap
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=palette), ax=ax, orientation='vertical')
        cbar.set_label('Target Distribution')
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]
        heatmap_img = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=palette, alpha=alpha)
        
        if use_sliders:
            filter_sliders = {}
            multiplier_sliders = {}
            for key in self.heatmaps.keys():
                y_position = 0+ 0.05 * list(self.heatmaps.keys()).index(key)
                
                ax_filter_slider = plt.axes([0.25, y_position, 0.25, 0.03])
                ax_multiplier_slider = plt.axes([0.60, y_position, 0.25, 0.03])
                
                filter_sliders[key] = Slider(ax_filter_slider, f'{key}:  σ', 0.0, 10.0, valinit=3, valstep=0.1)
                multiplier_sliders[key] = Slider(ax_multiplier_slider, 'α', 0.0, 3.0, valinit=1, valstep=0.1)
            # Save button
            save_ax = plt.axes([0.5, 0.9, 0.1, 0.04])
            save_button = plt.Button(save_ax, 'Save', color='white', hovercolor='0.975')

            def update(val):
                sigma_features = {key: slider.val for key, slider in filter_sliders.items()}
                alpha_features = {key: multiplier_sliders[key].val for key in filter_sliders.keys()}
                combined_heatmap = self.get_combined_heatmap(sigma_features, alpha_features)
                
                heatmap_img.set_data(combined_heatmap.T)
                plt.draw()

            def save(event):
                sigma_features = {key: slider.val for key, slider in filter_sliders.items()}
                alpha_features = {key: multiplier_sliders[key].val for key in filter_sliders.keys()}
                combined_heatmap = self.get_combined_heatmap(sigma_features, alpha_features)
                # Normalize the heatmap to the range 0-255
                combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min()) * 255
                
                temp_heatmap = np.flipud(combined_heatmap.T) # Makes sure that the map is oriented correctly
                
                height, width = temp_heatmap.shape
                greyscale_with_alpha = np.zeros((height, width, 2), dtype=np.uint8)
                # Set the grayscale channel based on the matrix values
                greyscale_with_alpha[..., 0] = temp_heatmap  # Greyscale (intensity) values
                # Set the alpha channel: 255 where matrix > 0, otherwise 0 (transparent)
                greyscale_with_alpha[..., 1] = np.where(temp_heatmap > -1, 255, 0)
                
                # Convert to a greyscale image with alpha
                img = Image.fromarray(greyscale_with_alpha, mode='LA')
                # Create a filename with the slider values
                slider_values = "_".join([f"{key}_sigma{filter_sliders[key].val:.1f}_alpha{multiplier_sliders[key].val:.1f}" for key in filter_sliders.keys()])
                filename = f"data/heatmaps/heatmap_{slider_values}.png"
                
                # Save the image with the generated filename
                img.save(filename)
                print("Heatmap saved as heatmap.png")

            save_button.on_clicked(save)
            
            for slider in filter_sliders.values():
                slider.on_changed(update)
            for slider in multiplier_sliders.values():
                slider.on_changed(update)
        
        plt.axis("equal")
        ax.set_axis_off()

        if export:
            plt.savefig("data/plots/plot.png",bbox_inches='tight')
        plt.show()
    
if __name__ == "__main__":
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    env = EnvironmentBuilder().set_polygon_file(polygon_file).set_feature(
        "wetlands", {"natural": ["water", "wetland"]}
    ).set_feature(
        "roads",
        {
            "highway": [
                "service",
                "track",
                "highway",
                "primary",
                "secondary",
                "tertiary",
                "residential",
            ]
        },
    ).build()

    env.interactive_plot()