from typing import List, Tuple
import math
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
import csv
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import concurrent.futures
import threading
       
import contextily as cx
from trajgenpy import Utils
import seaborn as sns


def plot_information_gain_histogram(dataset_dir):
    mean_length = 100*4  # Each iteration consists of 4 steps (ODE discretisation)
    # variance_length = 20  # Define the variance of the task lengths
    variance_length = (20 / 2) ** 2
    sensor_range = 10 # TODO change this to proper values
    sensor_variance = 2
    datasets = {}
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            pkl_file = os.path.join(folder_path, f"{folder}.pkl")
            if os.path.isfile(pkl_file):
                with open(pkl_file, 'rb') as f:
                    datasets[folder_path] = pickle.load(f)

    information_gains = []
    for path, experiment in datasets.items():
        experiment: ExperimentData
        for task in experiment.tasks:
            task: Task.TrajectoryTask
            task_length = np.random.normal(mean_length, variance_length)
            point_list = list(task.trajectory.coords)[:int(task_length)]
            information_gains.append(information_gain_from_points(point_list, experiment.heatmap, sensor_range, sensor_variance))

    plt.hist(information_gains, bins=30, edgecolor='black')
    plt.title('Information Gain per Task')
    plt.xlabel('Information Gain')
    plt.ylabel('Frequency')
    plt.show()


def image_to_world(x, y, meters_per_bin, minx, miny, buffer):
    x = x* meters_per_bin + minx - buffer
    y = y* meters_per_bin + miny - buffer
    return x, y

def world_to_image(x, y, meters_per_bin, minx, miny, buffer):
    x = (x - minx + buffer) / meters_per_bin
    y = (y - miny + buffer) / meters_per_bin
    return int(x), int(y)#Figure out if this rounding is bad

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

def get_filter_sigma(filter_width_meters, env):
    filter_width_pixels = filter_width_meters / env.meter_per_bin
    
    # Calculate the sigma value for the Gaussian filter
    sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
    return sigma


def task_allocation(boundary, tasks, agent_list):
    # Normalize the geoms
    cp = CoverageProblem.CoverageProblem(
        restricted_areas=None,
        search_area=boundary,
        tasks=tasks,
    )

    exp = Experiment.Runner(coverage_problem=cp, agents=agent_list)

    exp.solve(profiling_enabled=False)
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
    

def generate_dataset(environment_file, n_trajectories, steps,experiment_dir, generate_all=True, common_depot=False):
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
            paths = compute_trajectories(combined_heatmap, n_trajectories,steps, experiment_dir=f"data/DemaScenariosTasks/{environment_file}_{i}/", common_depot=common_depot)

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



def information_gain_from_points(points, heatmap, sensor_range, sensor_sigma):
    x, y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    # distances = np.min([np.sqrt((x - p[1])**2 + (y - p[0])**2) for p in points], axis=0)
    distances = np.min([np.linalg.norm(np.array([x - p[1], y - p[0]]), axis=0) for p in points], axis=0)
    # distances = np.min([np.hypot(x - p[1], y - p[0]) for p in points], axis=0)
    # Gaussian sensor model (decay with distance)
    sensor_model = np.exp(-distances**2 / (2 * sensor_sigma**2))
    sensor_mask = distances <= sensor_range
    information_gain = heatmap * sensor_model * sensor_mask
    return information_gain.sum() / heatmap.sum()

def information_gain_from_point(point, heatmap, sensor_range, sensor_sigma):
    x, y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    distances = np.sqrt((x - point[1])**2 + (y - point[0])**2)
    sensor_model = np.exp(-distances**2 / (2 * sensor_sigma**2))
    sensor_mask = distances <= sensor_range
    information_gain = heatmap* sensor_model * sensor_mask
    return information_gain.sum() / heatmap.sum()

@dataclass
class EvaluationResult:
    survivors_location: List[Tuple[float,float]]
    survivors_found_location: List[Tuple[float,float]]
    survivors_found_time: List[float]
    information_time: List[float]
    local_information_gain: List[float]
    global_information_gain: List[float]
    number_of_agents: int
    task_length_mean: float
    task_length_variance: float
    sensor_range: float
    sensor_variance: float
    agent_capacity:float

def evaluate_experiment(heatmap, trajectories, sensor_range, sensor_variance,task_length_mean,task_length_variance,number_of_agents,agent_capacity, experiment_data: ExperimentData):
    survivors = sample_points(heatmap, 100)
    survivors_world_coords = [image_to_world(p[1], p[0], experiment_data.meter_per_bin, experiment_data.min_x, experiment_data.min_y, experiment_data.buffer) for p in survivors]
    survivors_world_coords_copy = survivors_world_coords.copy()
    # results
    survivors_found_time =[]
    survivors_found_location =[]
    information_time = []
    local_information_gain_list = []
    global_information_gain_list = []
    global_information_gain =0
    
    x, y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    global_sensor_mask = np.zeros_like(heatmap, dtype=bool)
    
    # Make sure that the information gain is calculated correctly
    interpolated_trajectories = []
    for trajectory in trajectories:
        interpolated_trajectory = []
        for i in range(1, len(trajectory)):
            interpolated_trajectory.append(trajectory[i-1])
            distance = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
            if distance > 0.5 * sensor_range:
                num_interpolations = int(distance // (0.5 * sensor_range))
                for j in range(1, num_interpolations + 1):
                    interpolated_point = np.array(trajectory[i-1]) + (np.array(trajectory[i]) - np.array(trajectory[i-1])) * (j / (num_interpolations + 1))
                    interpolated_trajectory.append(interpolated_point)
        interpolated_trajectory.append(trajectory[-1])
        interpolated_trajectories.append(interpolated_trajectory)
    trajectories = interpolated_trajectories
    
    for trajectory in interpolated_trajectories:
        timer = 0
        local_sensor_mask = np.zeros_like(heatmap, dtype=bool)
        local_information_gain = 0

        for i in range(0, len(trajectory)):
            # Calculate the time it takes to traverse the trajectory
            agent_position = np.array(trajectory[i])

            if i == 0:
                distance = 0
            else:
                distance = np.linalg.norm(agent_position - np.array(trajectory[i-1]))
            
            time_to_traverse = distance / 5  # Assuming constant velocity of 5 m/s
            timer += time_to_traverse
            for survivor in survivors_world_coords:
                if np.linalg.norm(agent_position - np.array(survivor)) <= sensor_range:
                    survivors_found_time.append(timer)
                    survivors_found_location.append(survivor)
                    survivors_world_coords.remove(survivor)
                    
            # Calculate the information gain at each timestep:
            point_in_heatmap_coords = world_to_image(trajectory[i][0], trajectory[i][1], experiment_data.meter_per_bin, experiment_data.min_x, experiment_data.min_y, experiment_data.buffer)
            sensor_distances = np.sqrt((x - point_in_heatmap_coords[1])**2 + (y - point_in_heatmap_coords[0])**2)
            
            # Global information gain
            global_sensor_mask += sensor_distances <= sensor_range
            temp_information_gain = (heatmap*global_sensor_mask).sum()
            global_information_gain_list.append((temp_information_gain - global_information_gain) / heatmap.sum())
            global_information_gain = temp_information_gain
            
            # Local information gain
            local_sensor_mask += sensor_distances <= sensor_range
            temp_information_gain = (heatmap*local_sensor_mask).sum()
            local_information_gain_list.append((temp_information_gain - local_information_gain) / heatmap.sum())
            local_information_gain = temp_information_gain
            
            information_time.append(timer)
        
    # Sort the information time to ensure that the information gain is correctly ordered
    information_time = np.array(information_time)
    sorted_indices = np.argsort(information_time)
    information_time = information_time[sorted_indices]
    
    sorted_local_information_gain = np.array(local_information_gain_list)
    local_information_gain_list = sorted_local_information_gain[sorted_indices]

    sorted_global_information_gain = np.array(global_information_gain_list)
    global_information_gain_list = sorted_global_information_gain[sorted_indices]
    
    return EvaluationResult(survivors_location=survivors_world_coords_copy,
                            survivors_found_location=survivors_found_location, 
                            survivors_found_time=survivors_found_time, 
                            information_time=information_time, 
                            local_information_gain=local_information_gain_list, 
                            global_information_gain=global_information_gain_list,
                            number_of_agents=number_of_agents,
                            task_length_mean=task_length_mean,
                            task_length_variance=task_length_variance,
                            sensor_range=sensor_range,
                            sensor_variance=sensor_variance,
                            agent_capacity=agent_capacity
                            )


def plot_on_map(experiment_data, tasks, travel, eval, path, title):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    # Plot agent routes
    axs.ticklabel_format(style='plain', axis='both', useOffset=True, useMathText=True, scilimits=(0, 0))
    colors = plt.cm.get_cmap('tab10', len(tasks))
    
    for i, route in enumerate(tasks.values()):
        for segment in route:
            x, y = segment.xy
            axs.plot(x, y, linestyle='-', alpha=0.3, linewidth=10, color=colors(i))
            axs.plot(x, y, linestyle='-', alpha=1.0, color=colors(i))

    for i, route in enumerate(travel.values()):
        for segment in route:
            if len(segment) == 0:
                continue
            x, y = zip(*segment)
            axs.plot(x, y, linestyle=':', alpha=0.5, color=colors(i))
    show_survivors = False
    if show_survivors:
        for i, survivor in enumerate(eval.survivors_location):
            axs.plot(survivor[0], survivor[1], 'o', color='red')
        for i, survivor in enumerate(eval.survivors_found_location):
            axs.plot(survivor[0], survivor[1], 'o', color='green')
        
    Utils.plot_basemap(provider=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:2197")
            
    x, y = experiment_data.boundary.exterior.xy
    # axs.plot(x, y, color='black', linewidth=2, linestyle='--')
    # axs.axis('equal')
    minx, miny, maxx, maxy = experiment_data.boundary.bounds
    axs.set_xlim(minx, maxx)
    axs.set_ylim(miny, maxy)
    axs.set_axis_off()
    
    plt.tight_layout()
    # print("Saving plot:", path + title + ".png")
    plt.savefig(str(path) + str(title) + "allocation.png",bbox_inches='tight')
    
    # plt.show()
    plt.close()

def plot_result(show_survivors, experiment_data, tasks, travel,trajectories, eval, path,title, show=True):
    
    show_survivors = False
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot agent routes
    axs[0].ticklabel_format(style='plain', axis='both', useOffset=True, useMathText=True, scilimits=(0, 0))
    colors = plt.cm.get_cmap('tab10', len(tasks))

    # For hedac results
    # if not isinstance(tasks, dict):
    #     tasks = {i: [route.trajectory] for i, route in enumerate(tasks)}
    # if not isinstance(travel, dict):
    #     travel = {i: [segment] for i, segment in enumerate(travel)}
    # if not isinstance(trajectories, dict):
    #     trajectories = {i: [trajectory] for i, trajectory in enumerate(trajectories)}
    # for trajectory in trajectories.values():
    #     initial_position = trajectory[0][0]
    #     axs[0].plot(initial_position[0], initial_position[1], 'x', color='black', markersize=10)
    
    # For allocation results
    # for trajectory in trajectories.values():
    #     initial_position = trajectory[0]
    #     axs[0].plot(initial_position[0], initial_position[1], 'x', color='black', markersize=10)
    
    for i, route in enumerate(tasks.values()):
        for segment in route:
            x, y = segment.xy
            axs[0].plot(x, y, linestyle='-', alpha=0.3, linewidth=10, color=colors(i))
            axs[0].plot(x, y, linestyle='-', alpha=1.0, color=colors(i))

    for i, route in enumerate(travel.values()):
        for segment in route:
            if len(segment) == 0:
                continue
            x, y = zip(*segment)
            axs[0].plot(x, y, linestyle=':', alpha=0.5, color=colors(i))

    if show_survivors:
        for i, survivor in enumerate(eval.survivors_location):
            axs[0].plot(survivor[0], survivor[1], 'o', color='red')
        for i, survivor in enumerate(eval.survivors_found_location):
            axs[0].plot(survivor[0], survivor[1], 'o', color='green')
        
    x, y = experiment_data.boundary.exterior.xy
    axs[0].plot(x, y, color='black', linewidth=2, linestyle='--')
    axs[0].axis('equal')
    axs[0].set_xlabel('X Coordinate [m]')
    axs[0].set_ylabel('Y Coordinate [m]')
    axs[0].grid(True)

    # Plot survivors found over time
    times = eval.survivors_found_time
    sorted_times = sorted(times)
    axs[1].plot(sorted_times, range(1, len(sorted_times) + 1), '-', color='blue')
    axs[1].set_title('Survivors Found Over Time')
    axs[1].set_ylim(0, 100)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Number of Survivors Found')
    axs[1].grid(True)

    # Plot cumulative sum of local information gain over time
    information_time = np.array(eval.information_time)
    cumulative_local_information_gain = np.cumsum(eval.local_information_gain)
    
    axs[2].plot(information_time, cumulative_local_information_gain, '-', color='green')
    axs[2].set_title('Cumulative Sum of Local Information Gain Over Time')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Cumulative Local Information Gain [%]')
    axs[2].grid(True)
    
    # Plot cumulative sum of global information gain over time
    sorted_global_information_gain = np.array(eval.global_information_gain)
    cumulative_global_information_gain = np.cumsum(sorted_global_information_gain)
    
    axs[2].plot(information_time, cumulative_global_information_gain, '-', color='red', label='Global Information Gain')
    axs[2].legend(['Local Information Gain', 'Global Information Gain'])
    
    plt.tight_layout()
    # print("Saving plot:", path + title + ".png")
    plt.savefig(str(path) + str(title) + "results.png",bbox_inches='tight')
    
    # plt.show()
    plt.close()

def run_hedac_experiment(sensor_range, sensor_variance, task_variance, n_agents,task_length, common_depot, environment_file):
    environment_file = "FlatTerrainNature"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{environment_file}/{environment_file}_hedac_agents_{n_agents}_capacity_{task_length}_{timestamp}/"
    os.makedirs(experiment_dir, exist_ok=True)

    result_list = {}

    def process_experiment(j):
        result = {}
        start_time = datetime.datetime.now()
        experiment = generate_dataset("FlatTerrainNature", n_agents, task_length, experiment_dir + str(j) + "/", False, common_depot=common_depot)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Dataset generation for experiment {j} took {elapsed_time.total_seconds()} seconds")
        routes = [task.trajectory.coords for task in experiment.tasks]
        routes = [list([image_to_world(p[0], p[1], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer) for p in route]) for i, route in enumerate(routes)]
        
        eval = evaluate_experiment(experiment.heatmap,
                                   routes,
                                   sensor_range, 
                                   sensor_variance,
                                   task_length, 
                                   task_variance, 
                                   n_agents,
                                   task_length,  # agent capacity
                                   experiment)
        
        result["FlatTerrainNature" + str(j)] = {
            "heatmap": experiment.heatmap,
            "trajectories": routes,
            "travel": {},
            "tasks": experiment.tasks,
            "sensor_range": sensor_range,
            "sensor_variance": sensor_variance,
            "task_length_mean": task_length,
            "task_length_variance": 0,
            "number_of_agents": n_agents,
            "agent_capacity": task_length,
            "experiment_data": experiment,
            "survivors_location": eval.survivors_location,
            "survivors_found_location": eval.survivors_found_location,
            "survivors_found_time": eval.survivors_found_time,
            "information_time": eval.information_time,
            "local_information_gain": eval.local_information_gain,
            "global_information_gain": eval.global_information_gain,
            "evaluation": eval,
            "compute_time":elapsed_time.total_seconds()
        }

        # tasks = {i: [LineString(route)] for i, route in enumerate(routes)}
        # show_survivors = True
        # plot_result(show_survivors, experiment, tasks, {}, eval, experiment_dir, environment_file + str(j), show=False)
        return result
    mutex = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_experiment, j) for j in range(40)]
        for future in concurrent.futures.as_completed(futures):
            with mutex:
                result = future.result()
                print(f"Processing results from: {result.keys()}")
                result_list.update(result)
                print("results: ", result_list.keys())
                result_df = pd.DataFrame.from_dict(result_list, orient='index')
                result_df.to_pickle(os.path.join(experiment_dir, f"hedac_agents_{n_agents}_capacity_{task_length}.pkl"))

    result_df = pd.DataFrame.from_dict(result_list, orient='index')
    result_df.to_pickle(os.path.join(experiment_dir, f"hedac_agents_{n_agents}_capacity_{task_length}.pkl"))

        
def run_trajalloc_experiment(environment_file, title, number_of_agents, agent_capacity, sensor_range, sensor_variance, reward_shaping:callable, common_depot=True):
    # Comment this out to generate the dataset
    # generate_dataset(environment_file,40,200, True)
    # exit(0)
    # Load the dataset
    dataset_dir = "data/DemaScenariosTasks"
    datasets = {}
    experiment_name = environment_file + "_static_reward"
    
    # alter the tasks length to follow a mean and variance of a certain task length
    mean_length = 100*4  # Each iteration consists of 4 steps (ODE discretisation)
    variance_length = (20 / 2) ** 2
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{environment_file}/{environment_file}_allocation_{title}_agents_{number_of_agents}_capacity_{agent_capacity}_{timestamp}/"
    os.makedirs(experiment_dir, exist_ok=True)
    
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            pkl_file = os.path.join(folder_path, f"{folder}.pkl")
            if os.path.isfile(pkl_file):
                with open(pkl_file, 'rb') as f:
                    datasets[folder_path] = pickle.load(f)
    
    result_list = {}
    for path, experiment in datasets.items():
        experiment: ExperimentData
        experiment_name = os.path.basename(path)
        def process_task(task):
            task: Task.TrajectoryTask
            task_length = np.random.normal(mean_length, variance_length)
            point_list = list(task.trajectory.coords)[:int(task_length)]
            task.reward = information_gain_from_points(point_list, experiment.heatmap, sensor_range, sensor_variance)
            # task.reward = math.exp(task.reward)
            task.reward = reward_shaping(task.reward)
            point_list = [image_to_world(p[0], p[1], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer) for p in point_list]
            task.trajectory = LineString(point_list)
            task.__post_init__()
            return task

        with concurrent.futures.ThreadPoolExecutor() as executor:
            updated_tasks = list(executor.map(process_task, experiment.tasks))
 
        experiment.tasks = updated_tasks
        
        # The total reward will most likely be above 100. This leads to a optimistic reward...
        total_reward = sum(task.reward for task in experiment.tasks)
        print(f"Total reward: {total_reward}")
        
        if common_depot:
            initial =sample_points(experiment.heatmap, 1)[0]
            initial = image_to_world(initial[1], initial[0], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer)
            agent_list = [
                Agent.config(id, (initial[0] + np.random.normal(0, 2), initial[1] + np.random.normal(0, 2)), capacity=agent_capacity, max_velocity=10)
                for id in range(number_of_agents)
            ]
        else:
            points = sample_points(experiment.heatmap, number_of_agents)
            points = [image_to_world(p[0], p[1], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer) for p in points]
            points = sample_points(experiment.heatmap, number_of_agents)
            agent_list = [
                Agent.config(id,p, capacity=agent_capacity, max_velocity=5)
                for p in points
            ]

        start_time = datetime.datetime.now()
        allocation = task_allocation(experiment.boundary, experiment.tasks,agent_list)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Task allocation for experiment {experiment_name} took {elapsed_time.total_seconds()} seconds")
        
        eval= evaluate_experiment(heatmap=experiment.heatmap,
                                  trajectories=list(allocation.routes.values()),
                                  sensor_range=sensor_range, 
                                  sensor_variance=sensor_variance,
                                  task_length_mean=mean_length,
                                  task_length_variance=variance_length,
                                  number_of_agents=number_of_agents,
                                  agent_capacity=agent_capacity,
                                  experiment_data=experiment)
        
        result_list[experiment_name] = {
            "heatmap": experiment.heatmap,
            "trajectories": allocation.routes,
            "travel": allocation.transport,
            "tasks": allocation.tasks,
            "sensor_range": sensor_range,
            "sensor_variance": sensor_variance,
            "task_length_mean": mean_length,
            "task_length_variance": variance_length,
            "number_of_agents": number_of_agents,
            "agent_capacity": agent_capacity,
            "experiment_data": experiment,
            "survivors_location": eval.survivors_location,
            "survivors_found_location": eval.survivors_found_location,
            "survivors_found_time": eval.survivors_found_time,
            "information_time": eval.information_time,
            "local_information_gain": eval.local_information_gain,
            "global_information_gain": eval.global_information_gain,
            "evaluation": eval,      
            "compute_time":elapsed_time.total_seconds()
        }
        # plot_result(True, experiment,allocation.tasks, allocation.transport,eval, experiment_dir,experiment_name, show=False)

        result_df = pd.DataFrame.from_dict(result_list, orient='index')
        result_df.to_pickle(os.path.join(experiment_dir, f"{timestamp}.pkl"))
    
    # FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
    # HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
    # Urban: Parks, roads, pathways, Lakes, river, wetlands
    # Water: Wetlands, banks, roads

def evaluate_results(experiment_file):
    experiment_dir = os.path.dirname(experiment_file)
    df = pd.read_pickle(experiment_file)
    global_information_samples = []
    information_time_samples = []
    print("Evaluating results for:" + experiment_file)
    for row in df.iterrows():
        # print(f"Evaluating experiment: {row[0]}")
        # plot_result(
        #     show_survivors=True,
        #     experiment_data=row[1]['experiment_data'],
        #     tasks=row[1]['tasks'],
        #     travel=row[1]['travel'],
        #     trajectories=row[1]['trajectories'],
        #     eval=row[1]['evaluation'],
        #     path=experiment_dir + "/",
        #     title=row[0],
        #     show=False
        # )
        information_time_samples.append(row[1]["information_time"])
        global_information_samples.append(row[1]["global_information_gain"])

    # Compute the cumulative information gain for each sample
    cumulative_information_samples = [np.cumsum(global_info) for global_info in global_information_samples]

    # Find the mean and confidence interval for each time step
    # Pad the samples to the same length
    max_length = max(len(sample) for sample in cumulative_information_samples)
    padded_cumulative_information_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in cumulative_information_samples]
    mean_cumulative_information = np.mean(padded_cumulative_information_samples, axis=0)
    std_cumulative_information = np.std(padded_cumulative_information_samples, axis=0)
    confidence_interval = 1.96 * std_cumulative_information / np.sqrt(len(padded_cumulative_information_samples))

    # Plot the mean cumulative information gain with confidence interval
    plt.figure(figsize=(10, 5))
    # Pad the information_time_samples to the same length
    padded_information_time_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in information_time_samples]
    mean_information_time = np.mean(padded_information_time_samples, axis=0)
    plt.plot(mean_information_time, mean_cumulative_information, label='Mean Cumulative Information Gain')
    plt.fill_between(mean_information_time, 
                     mean_cumulative_information - confidence_interval, 
                     mean_cumulative_information + confidence_interval, 
                     color='b', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Time [s]')
    plt.ylabel('Cumulative Information Gain [%]')
    # plt.title('Mean Cumulative Information Gain Over Time with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(experiment_dir + "/mean_cumulative_information_gain.png")
    # plt.show()
    plt.close()

    # Plot the mean number of survivors found over time with confidence interval
    all_survivors_found_time = []
    for row in df.iterrows():
        all_survivors_found_time.append(row[1]["survivors_found_time"])

    for num_survivors in [5, 20, 50, 75]:
        detection_times = [sorted(times)[:num_survivors] for times in all_survivors_found_time if len(times) >= num_survivors]
        detection_times = [times[-1] for times in detection_times]
        mean_detection_time = np.mean(detection_times)
        std_detection_time = np.std(detection_times)
        confidence_interval_detection = 1.96 * std_detection_time / np.sqrt(len(detection_times))
        mean_detection_time, confidence_interval_detection
        print(f"Mean time for detecting the first {num_survivors} survivors: {mean_detection_time:.2f} seconds")
        print(f"95% confidence interval: ±{confidence_interval_detection:.2f} seconds")
    # for time in [100, 500, 1000]:
    #     cumulative_information_at_time=mean_cumulative_information[np.searchsorted(mean_information_time, time)], confidence_interval[np.searchsorted(mean_information_time, time)]
    #     print(f"Mean cumulative information gain at {time} seconds: {cumulative_information_at_time:.2f} ± {ci:.2f}")

def no_shaping(reward):
    return reward

def exponential_shaping(reward):
    return math.exp(reward)

def static_reward_shaping(reward):
    return 1

def run_allocation_experiments():
    sensor_range = 10
    sensor_variance = 2
    task_variance = 0
    common_depot = True
    total_budget = 8000
    # Task generation budget = 200*40= 8000, Generate tasks that are 1/2 the length of the budget to match the trajallocation sampling of trajectories
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment("FlatTerrainNature", "information_reward", n_agents,budget, sensor_range, sensor_variance, no_shaping, common_depot)
        
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment("FlatTerrainNature", "exponential_shaping", n_agents,budget, sensor_range, sensor_variance, exponential_shaping, common_depot)
    
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment("FlatTerrainNature", "static_reward", n_agents,budget, sensor_range, sensor_variance, static_reward_shaping, common_depot)

def plot_comparrative_results(*experiment_files,legends =[]):
    plt.figure(figsize=(10, 10))
    for i, experiment_file in enumerate(experiment_files):
        df = pd.read_pickle(experiment_file)
        global_information_samples = []
        information_time_samples = []
        for row in df.iterrows():
            information_time_samples.append(row[1]["information_time"])
            global_information_samples.append(row[1]["global_information_gain"])

        cumulative_information_samples = [np.cumsum(global_info) for global_info in global_information_samples]
        max_length = max(len(sample) for sample in cumulative_information_samples)
        padded_cumulative_information_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in cumulative_information_samples]
        mean_cumulative_information = np.mean(padded_cumulative_information_samples, axis=0)
        std_cumulative_information = np.std(padded_cumulative_information_samples, axis=0)
        confidence_interval = 1.96 * std_cumulative_information / np.sqrt(len(padded_cumulative_information_samples))

        padded_information_time_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in information_time_samples]
        mean_information_time = np.mean(padded_information_time_samples, axis=0)
        plt.plot(mean_information_time, mean_cumulative_information, label=legends[i])
        plt.fill_between(mean_information_time, 
                         mean_cumulative_information - confidence_interval, 
                         mean_cumulative_information + confidence_interval, 
                         alpha=0.2)
        

    plt.xlabel('Time [s]')
    plt.ylabel('Cumulative Information Gain [%]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/plots/comparative_cumulative_information_gain.png")
    plt.show()
    

def plot_information_gathered():

    experiment_files = [
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_2_capacity_4000_20241113_125852/20241113_125852.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241113_131536/20241113_131536.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_133333/20241113_133333.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_5_capacity_1600_20241113_135052/20241113_135052.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["Exponential Reward Shaping", "Information Reward", "Static Reward", "HEDAC"]
    agent_counts = [2, 3, 4, 5]

    data = []
    timestamp_for_comparrison = 600
    for i, experiment_file in enumerate(experiment_files):
        global_information_samples = []
        information_time_samples = []
        df = pd.read_pickle(experiment_file)
        for row in df.iterrows():
            information_time_samples.append(row[1]["information_time"])
            global_information_samples.append(row[1]["global_information_gain"])

        cumulative_information_samples = [np.cumsum(global_info) for global_info in global_information_samples]
        max_length = max(len(sample) for sample in cumulative_information_samples)
        padded_cumulative_information_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in cumulative_information_samples]
        information_time = np.mean([np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in information_time_samples], axis=0)
        for sample in padded_cumulative_information_samples:
            information_gain_at = sample[np.searchsorted(information_time, timestamp_for_comparrison)]
            category = categories[i // 4]
            agent_count = agent_counts[i % 4]
            data.append([agent_count, category, information_gain_at])
    df = pd.DataFrame(data, columns=["Agents", "Method", "Information gain"])
    
    plt.figure(figsize=(10, 10))
    sns.set_palette("tab10")
    # sns.violinplot(x="Agents", y="Information gain", hue="Method", data=df)
    sns.boxplot(x="Agents", y="Information gain", hue="Method", data=df)
    plt.title("Mean Cumulative Information Gain at "+ str(timestamp_for_comparrison) + " seconds for Different Experiments")
    plt.ylabel("Information gain")
    plt.legend(title="Method")
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig("data/plots/boxplot_mean_cumulative_information_gain.png")
    # plt.show()

def plot_percent_survivors_detected():
    experiment_files = [
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_2_capacity_4000_20241113_125852/20241113_125852.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241113_131536/20241113_131536.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_133333/20241113_133333.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_5_capacity_1600_20241113_135052/20241113_135052.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["Exponential Reward Shaping", "Information Reward", "Static Reward", "HEDAC"]
    agent_counts = [2, 3, 4, 5]
    n_detections = 75    
    data = []
    for i, experiment_file in enumerate(experiment_files):
        df = pd.read_pickle(experiment_file)
        
        # Plot the mean number of survivors found over time with confidence interval
        all_survivors_found_time = []
        for row in df.iterrows():
            all_survivors_found_time.append(row[1]["survivors_found_time"])

        detection_times = [sorted(times)[:n_detections] for times in all_survivors_found_time if len(times) >= n_detections]
        detection_times = [times[-1] for times in detection_times]
        mean_detection_time = np.mean(detection_times)
        std_detection_time = np.std(detection_times)
        confidence_interval_detection = 1.96 * std_detection_time / np.sqrt(len(detection_times))
        mean_detection_time, confidence_interval_detection
        print(f'Experiment: {experiment_file}')
        print(f'agents: {agent_counts[i % 4]} method: {categories[i // 4]}')
        print(f"Mean time for detecting the first {n_detections} survivors: {mean_detection_time:.2f} seconds")
        print(f"95% confidence interval: ±{confidence_interval_detection:.2f} seconds")
        
        for detection_time in detection_times:
            category = categories[i // 4]
            agent_count = agent_counts[i % 4]
            data.append([agent_count, category, detection_time])
    df = pd.DataFrame(data, columns=["Agents", "Method", "Detection time"])
    
    plt.figure(figsize=(10, 10))
    sns.set_palette("tab10")
    # sns.pointplot(x="Agents", y="Detection time", hue="Method", data=df, dodge=True, join=False, ci="sd", markers=["o", "s", "D", "^"], capsize=.1)
    # sns.violinplot(x="Agents", y="Detection time", hue="Method", data=df)
    sns.boxplot(x="Agents", y="Detection time", hue="Method", data=df)
    plt.title(f"Mean time for detecting the first {n_detections} survivors for Different Experiments")
    plt.ylabel("Time [s]")
    plt.legend(title="Method")
    # plt.ylim(0, 1000)
    plt.tight_layout()
    plt.savefig("data/plots/boxplot_mean_detection_time.png")
    # plt.show()


if __name__ == '__main__':
    dataset_dir = "data/DemaScenariosTasks"
    environment_file = "FlatTerrainNature"
    # run_allocation_experiments()
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241113_120808/20241113_120808.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl")
    plot_information_gathered()
    plot_percent_survivors_detected()
    # plot_comparrative_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_083943/20241113_083943.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241111_214659/20241111_214659.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241111_150838/20241111_150838.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl"
    #                           ,legends=["Exponential Shaping","Information Reward","Static Reward","HEDAC"])
    
    plot_comparrative_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_133333/20241113_133333.pkl",
                            "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl"
                            ,legends=["itCBBA","HEDAC"])
    
    
    # generate_dataset(environment_file,40,200, True)
    # plot_information_gain_histogram(dataset_dir)
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241111_223622/20241111_223622.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241111_154436/20241111_154436.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241111_150055/20241111_150055.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl")
    # run_hedac_experiment(sensor_range, sensor_variance, task_variance, n_agents,task_length, common_depot, environment_file)
    