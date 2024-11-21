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
from trajallocpy import Agent, CoverageProblem, Experiment, Task
from dataclasses import dataclass
from typing import List
import pickle
from sklearn.metrics.pairwise import rbf_kernel
log = trajgenpy.Logging.get_logger()

import datetime
import HEDAC_basic
    
    
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


def image_to_world(x, y, meters_per_bin, minx, miny, buffer):
    x = x* meters_per_bin + minx - buffer
    y = y* meters_per_bin + miny - buffer
    return x, y

def world_to_image(x, y, meters_per_bin, minx, miny, buffer):
    x = (x - minx + buffer) / meters_per_bin
    y = (y - miny + buffer) / meters_per_bin
    return int(x), int(y)#Figure out if this rounding is bad


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
        self.features = {}        
        with open(self.polygon_file, "r") as f:
            data = json.load(f)
        coordinates = data["features"][0]["geometry"]["coordinates"][0]
        query_region = shapely.Polygon(coordinates)
        log.info("Query region bounds: %s", query_region.bounds)
        self.polygon = GeoPolygon(query_region).set_crs("EPSG:2197")
        self.area = self.polygon.geometry.area
        log.info("Area of the polygon: %s km²", self.area / 1e6)
        self.minx, self.miny, self.maxx, self.maxy = self.polygon.geometry.bounds
        num_bins_x = int((self.maxx - self.minx) * 1 / self.meter_per_bin)
        num_bins_y = int((self.maxy - self.miny) * 1 / self.meter_per_bin)
        log.info("Number of bins x: %i y: %i", num_bins_x, num_bins_y)
        self.xedges = np.linspace(self.minx - self.buffer, self.maxx + self.buffer, num_bins_x + 1)
        self.yedges = np.linspace(self.miny - self.buffer, self.maxy + self.buffer, num_bins_y + 1)
        print(f"Size of x edges: {len(self.xedges)}")
        print(f"Size of y edges: {len(self.yedges)}")

        for key in tags.keys():
            feature = query_features(
                GeoPolygon(query_region),
                tags[key],
            )
            
            feature_collection = [geom for feature in feature.values() for geom in feature.geoms]
            if all(isinstance(geom, shapely.geometry.Polygon) for geom in feature_collection):
                feature_geom = GeoMultiPolygon(feature_collection).set_crs("EPSG:2197")
            elif all(isinstance(geom, LineString) for geom in feature_collection):
                feature_geom = GeoMultiTrajectory(feature_collection).set_crs("EPSG:2197")
            else:
                raise ValueError("Invalid feature type in feature collection")
            self.heatmaps[key] = self.generate_heatmap(feature_geom.geometry, self.sample_distance)
            self.features[key] = feature_geom

    def image_to_world(self, x, y):
        return image_to_world(x, y, self.meter_per_bin, self.minx, self.miny, self.buffer)
    
    def world_to_image(self, x, y):
        return world_to_image(x, y, self.meter_per_bin, self.minx, self.miny, self.buffer)

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
        x_y_combinations = np.array(np.meshgrid(self.xedges, self.yedges)).T.reshape(-1, 2)
        for feature in geometry_collection.geoms:
            interpolated_points = []
            if isinstance(feature, LineString):
                line = feature
                interpolated_points = self.interpolate_line(line, sample_distance)
            elif isinstance(feature, shapely.geometry.Polygon):        
                if infill_geometries:
                    # Check if all combinations of xedges and yedges are inside the polygon
                    mask = np.array([feature.contains(shapely.geometry.Point(x, y)) for x, y in x_y_combinations])
                    interpolated_points.extend([shapely.geometry.Point(x, y) for (x, y), m in zip(x_y_combinations, mask) if m])
                else:
                    line = feature.exterior
                    interpolated_points = self.interpolate_line(line, sample_distance)
                    
            # No intersection between the points and the polygon/line
            if not interpolated_points: 
                continue
            x, y = zip(*[(point.x, point.y) for point in interpolated_points])
            
            temp_heatmap, _, _ = np.histogram2d(x, y, bins=(self.xedges, self.yedges))
            heatmap += temp_heatmap
            
        # Make sure that overlapping features are not counted multiple times in the histogram
        heatmap = np.clip(heatmap, 0, 1)
        # Normalize based on the number of bins occupied
        heatmap = heatmap / np.sum(heatmap) 
        return heatmap

    
    def get_combined_heatmap(self, sigma_features, alpha_features):
        heatmap = np.zeros((len(self.xedges) - 1, len(self.yedges) - 1))
        for key in self.heatmaps.keys():
            
            # c = np.zeros((len(self.xedges) - 1, len(self.yedges) - 1))
            # for i in range(len(self.xedges) - 1):
            #     for j in range(len(self.yedges) - 1):
            #         x_center = (self.xedges[i] + self.xedges[i + 1]) / 2
            #         y_center = (self.yedges[j] + self.yedges[j + 1]) / 2
            #         c[j, i] += np.exp(- ((self.heatmaps[key] - x_center) ** 2.0 + (self.heatmaps[key] - y_center) ** 2.0) / (2.0 * sigma_features[key] ** 2))
            # c+=heatmap*alpha_features[key]
            # TODO convert this to use a RBF instead
            heatmap+= gaussian_filter((self.heatmaps[key]) *alpha_features[key], sigma=sigma_features[key])
            # normalize the histograms with the number of bins occupied
            # heatmap += temp_heatmap / np.max(temp_heatmap)
            # apply a RBF kernel to the heatmap
            # rbf_kernel = np.exp(-0.5 * (self.heatmaps[key] / np.max(self.heatmaps[key]))**2 / sigma_features[key]**2)
            # heatmap += rbf_kernel * alpha_features[key]
        
        # Make sure the probabilities sum to 1
        # heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def interactive_plot(self, use_sliders=True, show_basemap = True, show_features=False, export=False):
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
        if show_features:
            self.polygon.plot(linestyle="--", facecolor="none", edgecolor="black")
            for key, feature in self.features.items():
                if isinstance(feature, GeoMultiPolygon):
                    feature.plot()
                elif isinstance(feature, GeoMultiTrajectory):
                    feature.plot(color="red", linestyle="--")

        if show_basemap:
            alpha = 0.4
            Utils.plot_basemap(provider=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:2197")
        
        colors = [
            (1, 1, 1),    # White
            (0, 0, 1),    # Blue
            (0, 0.5, 1),    # Blue
            (0, 1, 1),    # Cyan
            (0.7, 1, 0),    # Green
            (1, 1, 0),    # Yellow
            (1, 0.5, 0),  # Orange
            (1, 0, 0),    # Red
            (0.5, 0, 0),  # Dark red
        ]

        # Create the colormap
        custom_cmap = LinearSegmentedColormap.from_list("jet_fade_to_white", colors)
        # custom_cmap = "jet"
        
        # Use the custom colormap for the heatmap
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), ax=ax, orientation='vertical')
        cbar.mappable.set_clim(vmin=heatmap.min(), vmax=heatmap.max())
        cbar.set_label('Target Distribution')
        extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]]
        heatmap_img = plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=custom_cmap, alpha=alpha)
        if use_sliders:
            filter_sliders = {}
            multiplier_sliders = {}
            for key in self.heatmaps.keys():
                y_position = 0+ 0.05 * list(self.heatmaps.keys()).index(key)
                
                ax_filter_slider = plt.axes([0.25, y_position, 0.25, 0.03])
                ax_multiplier_slider = plt.axes([0.60, y_position, 0.25, 0.03])
                
                filter_sliders[key] = Slider(ax_filter_slider, f'{key}:  sigma', 0.0, 10.0, valinit=3, valstep=0.1)
                multiplier_sliders[key] = Slider(ax_multiplier_slider, 'alpha', 0.0, 10.0, valinit=1, valstep=0.1)
            # Save button
            save_ax = plt.axes([0.5, 0.9, 0.1, 0.04])
            save_button = plt.Button(save_ax, 'Save', color='white', hovercolor='0.975')

            def update(val):
                sigma_features = {key: slider.val for key, slider in filter_sliders.items()}
                alpha_features = {key: multiplier_sliders[key].val for key in filter_sliders.keys()}
                combined_heatmap = self.get_combined_heatmap(sigma_features, alpha_features)
                # for key in filter_sliders.keys():
                #     print(f"{key} - sigma: {filter_sliders[key].val}, alpha: {multiplier_sliders[key].val}")
                heatmap_img.set_data(combined_heatmap.T)
                cbar.mappable.set_clim(vmin=combined_heatmap.min(), vmax=combined_heatmap.max())
                heatmap_img.set_cmap(custom_cmap)
                plt.draw()

            def save(event):
                sigma_features = {key: slider.val for key, slider in filter_sliders.items()}
                alpha_features = {key: multiplier_sliders[key].val for key in filter_sliders.keys()}
                combined_heatmap = self.get_combined_heatmap(sigma_features, alpha_features)
                # Normalize the heatmap to the range 0-255
                # combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min()) * 127
                
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
                filename = f"data/plots/heatmap_{slider_values}.png"
                
                # Save the image with the generated filename
                img.save(filename)
                print("Heatmap saved as heatmap.png")
                
                # Save the plot itself as a png
                # Temporarily hide the sliders and buttons
                for slider in filter_sliders.values():
                    slider.ax.set_visible(False)
                for slider in multiplier_sliders.values():
                    slider.ax.set_visible(False)
                save_button.ax.set_visible(False)

                # Save the plot area
                plt.savefig("data/plots/plot.png", bbox_inches='tight')

                # Restore the visibility of sliders and buttons
                for slider in filter_sliders.values():
                    slider.ax.set_visible(True)
                for slider in multiplier_sliders.values():
                    slider.ax.set_visible(True)
                save_button.ax.set_visible(True)

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
    
def get_filter_sigma(filter_width_meters, env):
    filter_width_pixels = filter_width_meters / env.meter_per_bin
    
    # Calculate the sigma value for the Gaussian filter
    sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
    return sigma


@dataclass
class ExperimentData:
    heatmap: np.ndarray # Target distribution
    tasks: List[Task.TrajectoryTask]
    boundary: shapely.geometry.Polygon
    min_x: float
    min_y: float
    buffer: float
    meter_per_bin: float
    computation_time: float = 0.0

# TODO Add this function to the Environment modelling
def generate_dataset(environment_file,features,sigma_features,alpha_features, n_trajectories, steps,experiment_dir=None, generate_all=True, common_depot=False):
    # Create a new environment
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    base_path = "data/DemaScenarios/"
    # environment_file = "FlatTerrainNature"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_dir is None:
        experiment_dir = f"experiments/{environment_file}/{timestamp}/"
    env_builder = EnvironmentBuilder().set_polygon_file(base_path + environment_file + ".geojson").set_buffer(10)
    for key, feature in features.items():
        env_builder.set_feature(key, feature)
    env = env_builder.build()
    combined_heatmap = env.get_combined_heatmap(sigma_features, alpha_features)

    if generate_all:
        import concurrent.futures
        def process_experiment(i):
            start_time = datetime.datetime.now()
            # Compute the trajectories inside the environment
            paths = compute_trajectories(combined_heatmap, n_trajectories,steps, experiment_dir=f"data/DemaScenariosTasks/{environment_file}_{i}/", common_depot=common_depot)

            tasks = [Task.TrajectoryTask(i, LineString(paths[i]), reward=1) for i in range(len(paths))]

            end_time = datetime.datetime.now()
            # Create an instance of the data class
            experiment_data = ExperimentData(
                heatmap=combined_heatmap, 
                boundary=env.polygon.geometry,
                tasks=tasks, 
                min_x=env.minx, 
                min_y=env.miny, 
                buffer=env.buffer, 
                meter_per_bin=env.meter_per_bin,
                computation_time=(end_time - start_time).total_seconds()
            )
            print(f"Experiment {i} took {end_time - start_time}")
            # Save the instance to a pickle file
            print(f"Saving experiment data for {environment_file}_{i}")
            with open(f'data/DemaScenariosTasks/{environment_file}_{i}/{environment_file}_{i}.pkl', 'wb') as f:
                pickle.dump(experiment_data, f)
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_experiment, i) for i in range(40)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    print(f"Experiment failed with error: {e}")
                    traceback.print_exc()
    else:
        # Compute the trajectories inside the environment
        paths = compute_trajectories(combined_heatmap,n_trajectories, steps, experiment_dir=experiment_dir,common_depot=common_depot)

        # tasks = []
        # for i, point_list in enumerate(paths):
        #     task_reward = information_gain_from_points(point_list, combined_heatmap, 10, 2)
        #     temp_path = [image_to_world(p[0], p[1],env.meter_per_bin,env.minx,env.miny, env.buffer) for p in point_list]
        #     tasks.append(Task.TrajectoryTask(i, LineString(temp_path), reward=task_reward))
            
        tasks = [Task.TrajectoryTask(i, LineString(paths[i]), reward=1) for i in range(len(paths))]

        # Create an instance of the data class
        experiment_data = ExperimentData(
            heatmap=combined_heatmap, 
            boundary=env.polygon.geometry,
            tasks=tasks, 
            min_x=env.minx, 
            min_y=env.miny, 
            buffer=env.buffer, 
            meter_per_bin=env.meter_per_bin
        )
        return experiment_data


def compute_trajectories(image, n_trajectories,steps = 100, sensor_variance=2, experiment_dir: str = "experiments/", common_depot=False):

    image = np.array(image, dtype=np.float64).T
    image_height, image_width = image.shape[:2]
            
    test = HEDAC_basic.HEDAC_basic()

    test.method = 'hedac'
    test.results_dir = experiment_dir
    test.sigma_m = 1 # Envrionemtal variance, for smoothing environments
    test.sigma_c = sensor_variance# agent sensor footprint variance

    # test.method = 'smc'
    # test.results_dir = 'experiments/smc_full'
    # test.sigma_m = 1
    # test.sigma_c = 2

    test.X = np.arange(image.shape[1])
    test.Y = np.arange(image.shape[0])
    test.T = np.arange(steps)

    test.samples = image
    test.alpha = 1.0
    test.beta = 0.5
    test.gamma = 0.1
    test.va = 1 # Step size for the agents, Do not change this, as this is the same as changing the number of timesteps
    test.sigma_ac = 0.1
    # test.kappa = 0.1
    test.sourcefun = HEDAC_basic.difsource
    # logsource, difsource, difsquaredsource, divsource, fullcoveragecource generate_difpowersource(0.5) generate_divpowersource(power=2.0)

    test.outputStep = -1 #test.T.shape[0]-1 # Output the results at the last time step
    if common_depot:
        point = sample_points(image, 1)
        initial_positions = []
        for _ in range(n_trajectories):
            initial_positions.extend([(p[0] + np.random.normal(0, 2), p[1] + np.random.normal(0, 2)) for p in point])
        test.agents =initial_positions
    else:
        test.agents = sample_points(image, n_trajectories)
    
    # Use the function in the compute_trajectories function
    test.search()
    
    # Export the paths to a dict with agent number and the path
    paths = []
    for i, (xa, ya) in enumerate(zip(test.XA, test.YA)):
        # paths[i] = [env.image_to_world(x, y) for x, y in zip(xa, ya)]
        paths.append([[x, y] for x, y in zip(xa, ya)])
    return np.array(paths)

def sample_points(prob_dist, num_points):
    image_height, image_width = prob_dist.shape[:2]
    # Flatten the probability distribution and create a list of coordinates
    flat_prob_dist = (prob_dist/np.sum(prob_dist)).flatten()
    coordinates = [(i % image_width, i // image_width) for i in range(image_width * image_height)]

    # Sample agent locations based on the probability distribution
    return [(coordinates[i][0], coordinates[i][1]) for i in np.random.choice(len(flat_prob_dist), size=num_points, p=flat_prob_dist)]


def generate_flatnature_dataset():
    # sigma_wetland = get_filter_sigma(10, env)
    # sigma_roads = get_filter_sigma(30, env)
    sigma_features = {"roads": 3.0, "wetlands": 3.0}
    alpha_features = {"roads": 1, "wetlands": 0.5}
    features = {"wetlands":  {"natural": ["water", "wetland"]},
                "roads":{
                    "highway": [
                        "service",
                        "track",
                        "highway",
                        "primary",
                        "secondary",
                        "tertiary",
                        "residential",
                        ]
                    }
                }
    generate_dataset("FlatTerrainNature",features, sigma_features, alpha_features, 40, 200, "experiments/FlatTerrainNature/")

def generate_urban_dataset():
    # sigma_wetland = get_filter_sigma(10, env)
    # sigma_roads = get_filter_sigma(30, env)
    # Tunes using env.interactive_plot()
    sigma_features = {"roads": 2.5, "wetlands": 3.0}
    alpha_features = {"roads": 10, "wetlands": 1.1}
    features = {"wetlands":  {"natural": ["water", "wetland"]},
                "roads":{
                    "highway": [
                        "service",
                        "track",
                        "highway",
                        "primary",
                        "secondary",
                        "tertiary",
                        "residential",
                        ]
                    }
                }
    generate_dataset("Urban",features, sigma_features, alpha_features, 40, 200, "experiments/Urban/")

def generate_hillyterrainnature_dataset():
    # sigma_wetland = get_filter_sigma(10, env)
    # sigma_roads = get_filter_sigma(30, env)
    # Tunes using env.interactive_plot()
    sigma_features = {"forrest": 3.0, "tracks": 4.0}
    alpha_features = {"forrest": 1.0, "tracks": 5.0}
    features = {"forrest":  {"natural": ["wood", "wetland"]},
                "tracks":
                    {
                        "highway": [
                            "service",
                            "track",
                            "highway",
                            "primary",
                            "secondary",
                            "tertiary",
                            "residential",
                            "path",
                        ]
                    },
                }
    # TODO create a input which defines whether the environment should do infill of the heatmap or not
    generate_dataset("HillyTerrainNature",features, sigma_features, alpha_features, 40, 200, "experiments/HillyTerrainNature/")

def generate_water_dataset():
    # sigma_wetland = get_filter_sigma(10, env)
    # sigma_roads = get_filter_sigma(30, env)
    # Tunes using env.interactive_plot()
    sigma_features = {"coastline": 3.0, "wetland": 3.0, "tracks": 3.0}
    alpha_features = {"coastline": 4, "wetland": 1, "tracks": 0.5}
    features = {"coastline": {"natural": "coastline"},
                "wetland": {"natural": ["wetland","water"]},
                "tracks":
                    {
                        "highway": [
                            "service",
                            "track",
                            "highway",
                            "primary",
                            "secondary",
                            "tertiary",
                            "residential",
                            "path",
                        ]
                    },
                }
    generate_dataset("Water",features, sigma_features, alpha_features, 40, 200, "experiments/Water/")

# FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
# HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
# Urban: Parks, roads, pathways, Lakes, river, wetlands
# Water: Wetlands, banks, roads

if __name__ == "__main__":
    # generate_urban_dataset()
    # generate_flatnature_dataset()
    # generate_hillyterrainnature_dataset()
    # generate_water_dataset()
    # exit(0)
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    sigma_features = {"roads": 3.0, "wetlands": 3.0}
    alpha_features = {"roads": 1, "wetlands": 0.5}
    
    env = EnvironmentBuilder().set_polygon_file(polygon_file).set_buffer(50).set_feature(
        "roads",{
                    "highway": [
                        "service",
                        "track",
                        "highway",
                        "primary",
                        "secondary",
                        "tertiary",
                        "residential",
                        ]
                    }
    ).set_feature("wetlands",  {"natural": ["water", "wetland"]}).build()
    
    # heatmap = env.get_combined_heatmap(sigma_features, alpha_features)
    # plt.contourf(env.xedges[:-1], env.yedges[:-1], heatmap.T, levels=10, cmap="jet")
    # plt.contourf(env.xedges[:-1], env.yedges[:-1], heatmap.T, cmap="jet")
    # plt.colorbar(label='Heatmap Intensity')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Heatmap Topology')
    # plt.show()
    
    
    env.interactive_plot(use_sliders=True, show_basemap=True, show_features=False, export=True)
    