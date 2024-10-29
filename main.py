from typing import List

import shapely
import HEDAC_basic
import environment_modelling
import trajallocpy
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString
from trajallocpy import Agent, CoverageProblem, Experiment, Task, Utility
import random
import datetime
import pickle
from dataclasses import dataclass
import os
from scipy.interpolate import Rbf
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def compute_trajectories(image,steps = 100, experiment_dir: str = "experiments/"):

    image = np.array(image, dtype=np.float64).T
    image_height, image_width = image.shape[:2]
            
    test = HEDAC_basic.HEDAC_basic()

    test.method = 'hedac'
    test.results_dir = experiment_dir
    test.sigma_m = 1 # Envrionemtal variance, for smoothing environments
    test.sigma_c = 2 # agent sensor footprint variance

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

    # Normalize the image to create a probability distribution
    prob_dist = image

    # Flatten the probability distribution and create a list of coordinates
    flat_prob_dist = prob_dist.flatten()
    coordinates = [(i % image_width, i // image_width) for i in range(image_width * image_height)]

    # Sample agent locations based on the probability distribution
    num_agents = 30
    # Assign the sampled coordinates to the agents
    test.agents = [(coordinates[i][0], coordinates[i][1]) for i in np.random.choice(len(flat_prob_dist), size=num_agents, p=flat_prob_dist)]
    
    test.search()
    
    # Export the paths to a dict with agent number and the path
    paths = []
    for i, (xa, ya) in enumerate(zip(test.XA, test.YA)):
        # paths[i] = [env.image_to_world(x, y) for x, y in zip(xa, ya)]
        paths.append([[x, y] for x, y in zip(xa, ya)])
    return np.array(paths)

def get_filter_sigma(filter_width_meters, env):
    filter_width_pixels = filter_width_meters / env.meter_per_bin
    
    # Calculate the sigma value for the Gaussian filter
    sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
    return sigma


def task_allocation(boundary, tasks, n_agents=3, capacity=1000):
    # Normalize the geoms
    cp = CoverageProblem.CoverageProblem(
        restricted_areas=None,
        search_area=boundary,
        tasks=tasks,
    )
    # initial = cp.generate_random_point_in_problem().coords.xy
    # agent_list = [
    #     Agent.config(id, (initial[0][0] + random.uniform(-10, 10), initial[1][0] + random.uniform(-10, 10)), capacity=capacity, max_velocity=10)
    #     for id in range(n_agents)
    # ]
    agent_list = [
        Agent.config(id,cp.generate_random_point_in_problem().coords.xy, capacity=capacity, max_velocity=10)
        for id in range(n_agents)
    ]
    exp = Experiment.Runner(coverage_problem=cp, agents=agent_list)

    exp.solve(profiling_enabled=False)
    # Save the results in a csv file
    (
        computeTime,
        iterations,
        totalRouteLength,
        sumOfTaskLengths,
        totalRouteCosts,
        rewards,
        route_list,
        maxRouteCost,
    ) = exp.evaluateSolution()
    
    
    return exp



@dataclass
class ExperimentData:
    heatmap: np.ndarray # Target distribution
    tasks: List[Task.TrajectoryTask]
    boundary: shapely.geometry.Polygon
    min_x: float
    min_y: float
    buffer: float
    meter_per_bin: float
    

def generate_dataset(environment_file, generate_all=False):

    # Create a new environment
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    base_path = "data/DemaScenarios/"
    # environment_file = "FlatTerrainNature"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{environment_file}/{timestamp}/"
    env = environment_modelling.EnvironmentBuilder().set_polygon_file(base_path + environment_file + ".geojson").set_feature(
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
    ).set_buffer(0).build()

    # Parse these parameters from the func
    sigma_wetland = get_filter_sigma(10, env)
    sigma_roads = get_filter_sigma(30, env)

    sigma_features = {"roads": sigma_roads, "wetlands": sigma_wetland}
    alpha_features = {"roads": 1, "wetlands": 0.5}

    combined_heatmap = env.get_combined_heatmap(sigma_features, alpha_features)
    
    gen_dataset = True
    if gen_dataset:
        import concurrent.futures
        def process_experiment(i):
            # Compute the trajectories inside the environment
            paths = compute_trajectories(combined_heatmap, steps=200, experiment_dir=f"data/DemaScenariosTasks/{environment_file}_{i}/")

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
                # TODO Include the sensor model used in HEDAC
            )

            # Save the instance to a pickle file
            with open(f'data/DemaScenariosTasks/{environment_file}_{i}/{environment_file}_{i}.pkl', 'wb') as f:
                pickle.dump(experiment_data, f)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_experiment, i) for i in range(40)]
    else:
        # Compute the trajectories inside the environment
        paths = compute_trajectories(combined_heatmap, steps=100, experiment_dir=experiment_dir)

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

def information_gain_from_task(points, heatmap, sensor_range):
    sigma=2
    x, y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    distances = np.min([np.sqrt((x - p[1])**2 + (y - p[0])**2) for p in points], axis=0)
    # Gaussian sensor model (decay with distance)
    sensor_model = np.exp(-distances**2 / (2 * sigma**2))
    sensor_mask = distances <= sensor_range
    information_gain = heatmap * sensor_model * sensor_mask
    return information_gain.sum() / heatmap.sum()
    
    
    # Extract x and y coordinates from the point list
    x_coords, y_coords = zip(*point_list)

    # Create a mask for the points in point_list using a Gaussian filter
    mask = np.zeros_like(experiment_data.heatmap, dtype=np.float64)
    for x, y in zip(x_coords, y_coords):
        mask[int(x),int(y)] = 1.0

    mask = gaussian_filter(mask, sigma=1)

    # update the task trajectory and the reward based on the covered area
    # normalize the reward by the total information of the environment
    return np.sum(experiment_data.heatmap[mask > 0]) / np.sum(experiment_data.heatmap) 
    
        

if __name__ == '__main__':
    # Comment this out to generate the dataset
    # generate_dataset("FlatTerrainNature", True)
    # exit(0)
    # Load the dataset
    dataset_dir = "data/DemaScenariosTasks"
    datasets = {}

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            pkl_file = os.path.join(folder_path, f"{folder}.pkl")
            if os.path.isfile(pkl_file):
                with open(pkl_file, 'rb') as f:
                    datasets[folder_path] = pickle.load(f)
                    
    
    # convert all the coordinates in the paths to real world coordinates
    # world_coordinates = np.zeros_like(paths)
    # for i in range(len(paths)):
    #     agent_path = []
    #     for j in range(len(paths[i])):
    #         agent_path.append(env.image_to_world(paths[i][j][0], paths[i][j][1]))
    #     world_coordinates[i] = agent_path

    for path, experiment_data in datasets.items():
        experiment_data: ExperimentData
        # Convert the boundary to image coordinates? TODO remove this  
        # boundary = shapely.affinity.translate(experiment_data.boundary, xoff=-experiment_data.min_x, yoff=-experiment_data.min_y)
        # boundary = shapely.affinity.scale(boundary, xfact=1/experiment_data.meter_per_bin, yfact=1/experiment_data.meter_per_bin, origin=(0, 0))
        # alter the tasks length to follow a mean and variance of a certain task length
        mean_length = 100*4  # Each iteration consists of 4 steps (ODE discretisation)
        variance_length = 20  # Define the variance of the task lengths
        
        coverage = np.zeros_like(experiment_data.heatmap)
        grid_x, grid_y = np.meshgrid(np.arange(experiment_data.heatmap.shape[1]), np.arange(experiment_data.heatmap.shape[0]))

        for task in experiment_data.tasks:
            task: Task.TrajectoryTask
            task_length = np.random.normal(mean_length, variance_length)
            point_list = list(task.trajectory.coords)[:int(task_length)]
            task.reward = information_gain_from_task(point_list, experiment_data.heatmap, 10)
            
            def image_to_world(x, y, meters_per_bin, minx, miny, buffer):
                x = x* meters_per_bin + minx - buffer
                y = y* meters_per_bin + miny - buffer
                return x, y
    
            # TODO convert the point_list to world coordinates for the allocation to make sense
            point_list = [image_to_world(p[0], p[1],experiment_data.meter_per_bin,experiment_data.min_x,experiment_data.min_y, experiment_data.buffer) for p in point_list]
            task.trajectory = LineString(point_list)
            task.__post_init__()
        
        # The total reward will most likely be above 100. This leads to a optimistic reward...
        total_reward = sum(task.reward for task in experiment_data.tasks)
        print(f"Total reward: {total_reward}")
        
        exp_results = task_allocation(experiment_data.boundary, experiment_data.tasks, n_agents=3, capacity=3000)
        

        plot = True
        if plot:
            plt.figure(figsize=(10, 10))
            # colors =  [mcolors.to_rgb(c) for c in plt.cm.jet(np.linspace(0, 1, 256))]
            # palette = LinearSegmentedColormap.from_list("custom_flare", colors)
            # plt.imshow(coverage.T, cmap=palette, interpolation='nearest', origin='lower')
            # Offset the boundary by the minimum value

            plt.ticklabel_format(style='plain', axis='both',useOffset=True, useMathText=True, scilimits=(0,0))
            colors = plt.cm.get_cmap('tab10', len(exp_results.tasks))
            for i, route in enumerate(exp_results.tasks.values()):
                for segment in route:
                    x, y = segment.xy
                    plt.plot(x, y, linestyle='-', alpha=0.3, linewidth=10, color=colors(i))
                    plt.plot(x, y, linestyle='-', alpha=1.0, color=colors(i))

            for i, route in enumerate(exp_results.transport.values()):
                for segment in route:
                    if len(segment) == 0:
                        continue
                    x, y = zip(*segment)
                    plt.plot(x, y, linestyle=':', alpha=0.5,color=colors(i))
                    
            x, y = experiment_data.boundary.exterior.xy
            
            plt.plot(x, y, color='black', linewidth=2, linestyle='--')
            # env.polygon.plot()
            plt.title('Agent Routes')
            plt.axis('equal')
            plt.xlabel('X Coordinate [m]')
            plt.ylabel('Y Coordinate [m]')
            plt.grid(True)
            plt.savefig(path + "/task_allocation.png")
            plt.pause(0.1)
            plt.close()

    
    # TODO oct 17
    # Identify features for all the environments and define some sigma and alpha value to the scenarios, and document it as being static throughout the experiments
    # Generate trajectories using HEDAC
    # Identify how long the trajectories should be when having 30 tasks generated in hedac
    # Document this in the paper as a initial environment definition

    # Create a dataset of tasks and target distribution and save them to a file
    # Remove the extra reward for optimizing the return, this will create better paths for the agents
    
    # Dataset:
    # Create a dataset class, and export it as a pickle:
    # Routes/tasks --> TrajectoryTask in image coordinates, saved as a pickle
    # Heatmap --> numpy

    # Environment features: 
    # FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
    # HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
    # Urban: Parks, roads, pathways, Lakes, river, wetlands
    # Water: Wetlands, banks, roads