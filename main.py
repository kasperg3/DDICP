import concurrent.futures
import datetime
import math
import os
import pickle
import threading
from dataclasses import dataclass
import time
from typing import List, Tuple

import trajgenpy
import json
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from trajallocpy import Agent, CoverageProblem, Experiment, Task
import environment_modelling
from environment_modelling import world_to_image, image_to_world, ExperimentData, get_filter_sigma
from scipy.spatial import KDTree

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

def get_sensor_mask(points,heatmap, tree: KDTree, sensor_range):
    # Query the points within the sensor range
    indices = tree.query_ball_point(np.array(points), sensor_range)
    sensor_mask = np.zeros_like(heatmap, dtype=bool)
    for idx in indices:
        sensor_mask.ravel()[idx] = True
    return sensor_mask

def information_gain_from_points_tree(points,heatmap, tree: KDTree, sensor_range, sensor_sigma):
    sensor_mask = get_sensor_mask(points, heatmap, tree, sensor_range)
    # sensor_model = np.exp(-np.square(np.linalg.norm(np.array(np.where(sensor_mask)).T - np.array(points), axis=1)) / (2 * sensor_sigma**2))
    information_gain = heatmap * sensor_mask # *sensor_model

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
    survivors = environment_modelling.sample_points(heatmap, 100)
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
    # Create a KDTree for the heatmap points
    heatmap_points = np.column_stack((y.ravel(), x.ravel())) # inverted x and y to match the heatmap coordinates
    tree = KDTree(heatmap_points)
    
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
            # information_gain_from_points_tree(point_in_heatmap_coords, heatmap, tree, sensor_range, sensor_variance)
            # sensor_distances = np.sqrt((x - point_in_heatmap_coords[1])**2 + (y - point_in_heatmap_coords[0])**2)
            # sensor_mask = sensor_distances <= sensor_range
            sensor_range_in_bins = sensor_range / experiment_data.meter_per_bin
            sensor_mask = get_sensor_mask([point_in_heatmap_coords], heatmap, tree, sensor_range_in_bins)
            # Global information gain
            global_sensor_mask += sensor_mask
            temp_information_gain = (heatmap*global_sensor_mask).sum()
            global_information_gain_list.append((temp_information_gain - global_information_gain) / heatmap.sum())
            global_information_gain = temp_information_gain
            
            # Local information gain
            local_sensor_mask += sensor_mask
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

def run_hedac_experiment(sensor_range, features, sigma_features, alpha_features, sensor_variance, task_variance, n_agents,task_length, common_depot, environment_file):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{environment_file}/{environment_file}_hedac_agents_{n_agents}_capacity_{task_length}_{timestamp}/"
    os.makedirs(experiment_dir, exist_ok=True)

    result_list = {}

    def process_experiment(j):
        result = {}
        start_time = datetime.datetime.now()
        experiment = environment_modelling.generate_dataset(environment_file,features,sigma_features, alpha_features, n_agents, task_length, experiment_dir + str(j) + "/", False, common_depot=common_depot)
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
        
        result[environment_file + str(j)] = {
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
                result_df = pd.DataFrame.from_dict(result_list, orient='index')
                result_df.to_pickle(os.path.join(experiment_dir, f"hedac_agents_{n_agents}_capacity_{task_length}.pkl"))

def run_hedac_experiments(environment_file, features, sigma_features, alpha_features, experiment_dir):
    sensor_range = 10
    sensor_variance = 2
    common_depot = True
    total_budget = 8000
    for n_agents in [2, 3, 4, 5]:
        task_length = int(total_budget/n_agents)
        run_hedac_experiment(sensor_range,features, sigma_features,alpha_features, sensor_variance, 0, n_agents, task_length, common_depot, environment_file)

def run_trajalloc_experiment(environment_file, title, number_of_agents, agent_capacity, sensor_range, sensor_variance, reward_shaping:callable, common_depot=True):
    # Load the dataset
    dataset_dir = "data/DemaScenariosTasks/" + environment_file
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
        sensor_range_bins_units= sensor_range/experiment.meter_per_bin
        # Create a KDTree for the heatmap points
        x, y = np.meshgrid(np.arange(experiment.heatmap.shape[1]), np.arange(experiment.heatmap.shape[0]))
        heatmap_points = np.column_stack((y.ravel(), x.ravel())) # inverted x and y to match the heatmap coordinates
        heatmap_tree = KDTree(heatmap_points)
    
        experiment: ExperimentData
        experiment_name = os.path.basename(path)
        
        for task in experiment.tasks:
            task: Task.TrajectoryTask
            task_length = np.random.normal(mean_length, variance_length)
            point_list = list(task.trajectory.coords)[:int(task_length)]
            task.reward = information_gain_from_points_tree(point_list, experiment.heatmap, heatmap_tree, sensor_range_bins_units, sensor_variance)
            task.reward = reward_shaping(task.reward)
            point_list = [image_to_world(p[0], p[1], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer) for p in point_list]
            task.trajectory = LineString(point_list)
            task.__post_init__()
        
        # The total reward will most likely be above 100. This leads to a optimistic reward...
        total_reward = sum(task.reward for task in experiment.tasks)
        print(f"Total reward: {total_reward}")
        
        if common_depot:
            initial =environment_modelling.sample_points(experiment.heatmap, 1)[0]
            initial = image_to_world(initial[1], initial[0], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer)
            agent_list = [
                Agent.config(id, (initial[0] + np.random.normal(0, 2), initial[1] + np.random.normal(0, 2)), capacity=agent_capacity, max_velocity=10)
                for id in range(number_of_agents)
            ]
        else:
            points = environment_modelling.sample_points(experiment.heatmap, number_of_agents)
            points = [image_to_world(p[1], p[0], experiment.meter_per_bin, experiment.min_x, experiment.min_y, experiment.buffer) for p in points]
            agent_list = [
                Agent.config(id,p, capacity=agent_capacity, max_velocity=5)
                for p in points
            ]

        start_time = datetime.datetime.now()
        allocation = task_allocation(experiment.boundary, experiment.tasks,agent_list)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Task allocation for experiment {experiment_name} took {elapsed_time.total_seconds()} seconds")
        start_time = time.time()
        eval= evaluate_experiment(heatmap=experiment.heatmap,
                                  trajectories=list(allocation.routes.values()),
                                  sensor_range=sensor_range, 
                                  sensor_variance=sensor_variance,
                                  task_length_mean=mean_length,
                                  task_length_variance=variance_length,
                                  number_of_agents=number_of_agents,
                                  agent_capacity=agent_capacity,
                                  experiment_data=experiment)
        end_time = time.time()
        print(f"Experiment evaluation took {end_time - start_time} seconds")
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

def no_shaping(reward):
    return reward

def exponential_shaping(reward):
    return math.exp(reward*5)

def static_reward_shaping(reward):
    return 1

def run_allocation_experiments(environment_file):
    sensor_range = 10
    sensor_variance = 2
    common_depot = True
    total_budget = 8000
    # Task generation budget = 200*40= 8000, Generate tasks that are 1/2 the length of the budget to match the trajallocation sampling of trajectories
    budgets = [1000, 700, 500, 400]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for n_agents in [2, 3, 4, 5]:
            budget = int(total_budget/n_agents)
            futures.append(executor.submit(run_trajalloc_experiment, environment_file, "exponential_shaping", n_agents, budget, sensor_range, sensor_variance, exponential_shaping, common_depot))
            futures.append(executor.submit(run_trajalloc_experiment,environment_file, "static_reward", n_agents, budget, sensor_range, sensor_variance, static_reward_shaping, common_depot))
        for future in concurrent.futures.as_completed(futures):
            future.result()

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
            initial =environment_modelling.sample_points(experiment.heatmap, 1)[0]
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
    # Distance between sweeps should be 20 to match the trajectory generation
    # generate_sweep_tasks("data/DemaScenarios/FlatTerrainNature.geojson", 20)
    run_allocation_experiments("FlatTerrainNature")
    # run_allocation_experiments("HillyTerrainNature")
    # run_allocation_experiments("Urban")
    # run_allocation_experiments("Water")
    # generate_dataset(environment_file,40,200, True)
    exit(0)
    
    sigma_features = {"roads": 3, "wetlands": 3.0}
    alpha_features = {"roads": 0.5, "wetlands": 1}
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
    run_hedac_experiments("FlatTerrainNature", features, sigma_features, alpha_features,"experiments/FlatTerrainNature/")
