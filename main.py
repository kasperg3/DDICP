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
            
def image_to_world(x, y, meters_per_bin, minx, miny, buffer):
    x = x* meters_per_bin + minx - buffer
    y = y* meters_per_bin + miny - buffer
    return x, y

def world_to_image(x, y, meters_per_bin, minx, miny, buffer):
    x = (x - minx + buffer) / meters_per_bin
    y = (y - miny + buffer) / meters_per_bin
    return int(x), int(y)#Figure out if this rounding is bad

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


    
    # Use the function in the compute_trajectories function
    test.agents = sample_points(image, 30)
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
    
    if generate_all:
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

def information_gain_from_task(points, heatmap, sensor_range, sensor_sigma):
    x, y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    distances = np.min([np.sqrt((x - p[1])**2 + (y - p[0])**2) for p in points], axis=0)
    # Gaussian sensor model (decay with distance)
    sensor_model = np.exp(-distances**2 / (2 * sensor_sigma**2))
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
    
def evaluate_experiment(heatmap, trajectories, sensor_range, sensor_sigma):
    survivors = sample_points(heatmap, 100)
    survivors_world_coords = [image_to_world(p[1], p[0], experiment_data.meter_per_bin, experiment_data.min_x, experiment_data.min_y, experiment_data.buffer) for p in survivors]
    survivors_found =[]
    timer = 0
    for trajectory in trajectories:
        for i in range(1, len(trajectory)):
            # Calculate the time it takes to traverse the trajectory
            agent_position = np.array(trajectory[i])
            distance = np.linalg.norm(agent_position - np.array(trajectory[i-1]))
            time_to_traverse = distance / 5  # Assuming constant velocity of 5 m/s
            timer += time_to_traverse
            for survivor in survivors_world_coords:
                if np.linalg.norm(agent_position - np.array(survivor)) <= sensor_range:
                    survivors_found.append((timer, survivor))
                    survivors_world_coords.remove(survivor)

    # # Using the sensormodel, evaluate how much information was gained from the trajectories over the heatmap
    # trajectories_in_heatmap_coords = []
    # for trajectory in trajectories:
    #     trajectory_in_heatmap_coords = [world_to_image(p[0], p[1], experiment_data.meter_per_bin, experiment_data.min_x, experiment_data.min_y, experiment_data.buffer) for p in trajectory]
    #     trajectories_in_heatmap_coords.append(trajectory_in_heatmap_coords)
    
    # information_gain = 0
    # for trajectory in trajectories:
    #     information_gain += information_gain_from_task(trajectory, heatmap, sensor_range, sensor_sigma)

    # Using the heatmap, sample n_survivors=100, based on the probability distribution of the heatmap
    # Using the trajectories and the sensor model, calculate how many and how fast the different survivors were located.
    # - the agents that traverse the trajectories as simple point mass systems.
    # Using the sensormodel, evaluate how much information was gained from the trajectories over the heatmap
    
    plot = True
    show_survivors = True
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot agent routes
        axs[0].ticklabel_format(style='plain', axis='both', useOffset=True, useMathText=True, scilimits=(0, 0))
        colors = plt.cm.get_cmap('tab10', len(exp_results.tasks))
        
        for i, route in enumerate(exp_results.tasks.values()):
            for segment in route:
                x, y = segment.xy
                axs[0].plot(x, y, linestyle='-', alpha=0.3, linewidth=10, color=colors(i))
                axs[0].plot(x, y, linestyle='-', alpha=1.0, color=colors(i))

        for i, route in enumerate(exp_results.transport.values()):
            for segment in route:
                if len(segment) == 0:
                    continue
                x, y = zip(*segment)
                axs[0].plot(x, y, linestyle=':', alpha=0.5, color=colors(i))
        
        if show_survivors:
            for survivor in survivors_world_coords:
                axs[0].plot(survivor[0], survivor[1], 'ro')  # Plot survivors as red dots
            for time, survivor in survivors_found:
               axs[0].plot(survivor[0], survivor[1], 'o', color='green')
        
        x, y = experiment_data.boundary.exterior.xy
        axs[0].plot(x, y, color='black', linewidth=2, linestyle='--')
        axs[0].set_title('Agent Routes')
        axs[0].axis('equal')
        axs[0].set_xlabel('X Coordinate [m]')
        axs[0].set_ylabel('Y Coordinate [m]')
        axs[0].grid(True)

        # Plot survivors found over time
        times, _ = zip(*survivors_found)
        axs[1].plot(times,range(1, len(times) + 1), '-', color='blue')
        axs[1].set_title('Survivors Found Over Time')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Number of Survivors Found')
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(path + "/task_allocation_and_survivors_found.png")
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    # Comment this out to generate the dataset
    # generate_dataset("FlatTerrainNature", False)
    # exit(0)
    # Load the dataset
    dataset_dir = "data/DemaScenariosTasks"
    datasets = {}
    sensor_range = 10 # TODO change this to proper values
    sensor_variance = 2
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            pkl_file = os.path.join(folder_path, f"{folder}.pkl")
            if os.path.isfile(pkl_file):
                with open(pkl_file, 'rb') as f:
                    datasets[folder_path] = pickle.load(f)

    for path, experiment_data in datasets.items():
        experiment_data: ExperimentData
        # alter the tasks length to follow a mean and variance of a certain task length
        mean_length = 100*4  # Each iteration consists of 4 steps (ODE discretisation)
        variance_length = 20  # Define the variance of the task lengths
        
        coverage = np.zeros_like(experiment_data.heatmap)
        grid_x, grid_y = np.meshgrid(np.arange(experiment_data.heatmap.shape[1]), np.arange(experiment_data.heatmap.shape[0]))

        for task in experiment_data.tasks:
            task: Task.TrajectoryTask
            task_length = np.random.normal(mean_length, variance_length)
            point_list = list(task.trajectory.coords)[:int(task_length)]
            task.reward = information_gain_from_task(point_list, experiment_data.heatmap, sensor_range, sensor_variance)

            # TODO convert the point_list to world coordinates for the allocation to make sense
            point_list = [image_to_world(p[0], p[1],experiment_data.meter_per_bin,experiment_data.min_x,experiment_data.min_y, experiment_data.buffer) for p in point_list]
            task.trajectory = LineString(point_list)
            task.__post_init__()
        
        # The total reward will most likely be above 100. This leads to a optimistic reward...
        total_reward = sum(task.reward for task in experiment_data.tasks)
        print(f"Total reward: {total_reward}")
        
        exp_results = task_allocation(experiment_data.boundary, experiment_data.tasks, n_agents=3, capacity=2000)

        evaluate_experiment(experiment_data.heatmap,list(exp_results.routes.values()), sensor_range,sensor_variance)
        
    
    # Environment features: 
    # FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
    # HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
    # Urban: Parks, roads, pathways, Lakes, river, wetlands
    # Water: Wetlands, banks, roads