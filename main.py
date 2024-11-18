import concurrent.futures
import datetime
import math
import os
import pickle
import threading
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from trajallocpy import Agent, CoverageProblem, Experiment, Task
import environment_modelling
import HEDAC_basic


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

    exp = Experiment.Runner(coverage_problem=cp, agents=agent_list, enable_plotting=False)

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
    
# TODO Add this function to the Environment modelling
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

        result_df = pd.DataFrame.from_dict(result_list, orient='index')
        result_df.to_pickle(os.path.join(experiment_dir, f"{timestamp}.pkl"))
    
    # FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
    # HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
    # Urban: Parks, roads, pathways, Lakes, river, wetlands
    # Water: Wetlands, banks, roads

def no_shaping(reward):
    return reward

def exponential_shaping(reward):
    return math.exp(reward*5)

def static_reward_shaping(reward):
    return 1

def run_allocation_experiments(environment_file):
    sensor_range = 10
    sensor_variance = 2
    task_variance = 0
    common_depot = True
    total_budget = 8000
    # Task generation budget = 200*40= 8000, Generate tasks that are 1/2 the length of the budget to match the trajallocation sampling of trajectories
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment(environment_file, "information_reward", n_agents,budget, sensor_range, sensor_variance, no_shaping, common_depot)
        
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment(environment_file, "exponential_shaping", n_agents,budget, sensor_range, sensor_variance, exponential_shaping, common_depot)
    
    for n_agents in [ 2, 3, 4, 5]:
        budget = int(total_budget/n_agents)
        run_trajalloc_experiment(environment_file, "static_reward", n_agents,budget, sensor_range, sensor_variance, static_reward_shaping, common_depot)

import trajgenpy
import json
import matplotlib.pyplot as plt
def generate_sweep_tasks(polygon_file, distance_between_sweeps):
    print("Generating sweep tasks")
    base_path = "data/DemaScenarios/"
    environment_file = "FlatTerrainNature"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{environment_file}/{timestamp}/"
        
    with open(polygon_file, "r") as f:
        data = json.load(f)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    query_region = trajgenpy.Geometries.GeoPolygon(shapely.Polygon(coordinates))
    query_region = query_region.set_crs("EPSG:2197")
    
    polygons = trajgenpy.Geometries.decompose_polygon(query_region.geometry, None)
    task_geoms = []
    for polygon in polygons:
        task_geoms.extend(trajgenpy.Geometries.generate_sweep_pattern(polygon, distance_between_sweeps))
    
    # Create the heatmap: 
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
    
    # tasks = [Task.TrajectoryTask(i, task, reward=1) for i, task in enumerate(tasks)]

    # Parse these parameters from the func
    sigma_wetland = get_filter_sigma(10, env)
    sigma_roads = get_filter_sigma(30, env)

    sigma_features = {"roads": sigma_roads, "wetlands": sigma_wetland}
    alpha_features = {"roads": 1, "wetlands": 0.5}

    combined_heatmap = env.get_combined_heatmap(sigma_features, alpha_features)
    tasks = []
    sensor_range = 10
    for i, geom in enumerate(task_geoms):
        # interpolated_points = []
        # for i in range(1, len(geom.coords)):
        #     interpolated_points.append(geom.coords[i-1])
        #     distance = np.linalg.norm(np.array(geom.coords[i]) - np.array(geom.coords[i-1]))
        #     if distance > sensor_range/2:
        #         num_interpolations = int(distance // sensor_range)
        #         for j in range(1, num_interpolations + 1):
        #             interpolated_point = np.array(geom.coords[i-1]) + (np.array(geom.coords[i]) - np.array(geom.coords[i-1])) * (j / (num_interpolations + 1))
        #             interpolated_points.append(interpolated_point)
        # interpolated_points.append(geom.coords[-1])
        # geom = LineString(interpolated_points)
        # image_coords = [world_to_image(p[0], p[1], env.meter_per_bin, env.minx, env.miny, env.buffer) for p in geom.coords]
        reward =1# exponential_shaping(information_gain_from_points(image_coords, combined_heatmap, sensor_range, 2))
        task = Task.TrajectoryTask(i, geom, reward=reward)
        tasks.append(task)
    experiment = ExperimentData(combined_heatmap,tasks,env.polygon.geometry,env.minx,env.miny,env.buffer,env.meter_per_bin)
    with open(f'data/SweepTasks/{environment_file}_sweep_tasks.pkl', 'wb') as f:
        pickle.dump(experiment, f)

    sensor_variance = 2
    mean_length = 100*4  # Each iteration consists of 4 steps (ODE discretisation)
    result_list ={}
    for number_of_agents in [2,3,4,5]:
        agent_capacity = int(8000/number_of_agents)
        experiment_dir = "experiments/" + environment_file +"_sweep_tasks_agents_" + str(number_of_agents) + "_capacity_" + str(agent_capacity)
        for i in range(40): 
            experiment_name = environment_file + "_sweep_tasks_" + str(i)
            initial =sample_points(experiment.heatmap, 1)[0]
            initial = image_to_world(initial[1], initial[0], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer)
            agent_list = [
                Agent.config(id, (initial[0] + np.random.normal(0, 2), initial[1] + np.random.normal(0, 2)), capacity=agent_capacity, max_velocity=10)
                for id in range(number_of_agents)
            ]

            start_time = datetime.datetime.now()
            allocation = task_allocation(experiment.boundary, tasks,agent_list)
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            print(f"Task allocation for experiment {experiment_name} took {elapsed_time.total_seconds()} seconds")
            
            eval= evaluate_experiment(heatmap=experiment.heatmap,
                                        trajectories=list(allocation.routes.values()),
                                        sensor_range=sensor_range, 
                                        sensor_variance=sensor_variance,
                                        task_length_mean=0,
                                        task_length_variance=0,
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
                "task_length_variance": 0,
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

        result_df = pd.DataFrame.from_dict(result_list, orient='index')
        os.makedirs(experiment_dir, exist_ok=True)
        result_df.to_pickle(os.path.join(experiment_dir, f"{timestamp}.pkl"))


if __name__ == '__main__':
    dataset_dir = "data/DemaScenariosTasks"
    environment_file = "FlatTerrainNature"
    # Distance between sweeps should be 20 to match the trajectory generation
    # generate_sweep_tasks("data/DemaScenarios/FlatTerrainNature.geojson", 20)
    run_allocation_experiments(environment_file)
    # generate_dataset(environment_file,40,200, True)
    # run_hedac_experiment(sensor_range, sensor_variance, task_variance, n_agents,task_length, common_depot, environment_file)
    