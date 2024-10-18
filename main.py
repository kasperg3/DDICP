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
import matplotlib.pyplot as plt

def compute_trajectories(env: environment_modelling.Environment, experiment_dir: str = "experiments/"):
    # Parse these parameters from the func
    sigma_wetland = get_filter_sigma(10, env)
    sigma_roads = get_filter_sigma(30, env)

    sigma_features = {"roads": sigma_roads, "wetlands": sigma_wetland}
    alpha_features = {"roads": 1, "wetlands": 0.5}

    image = env.get_combined_heatmap(sigma_features, alpha_features)
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
    test.T = np.arange(100)

    test.samples = image
    test.alpha = 1.0
    test.beta = 0.5
    test.gamma = 0.1
    test.va = 1 # Step size for the agents, Do not change this, as this is the same as changing the number of timesteps
    test.sigma_ac = 0.1
    # test.kappa = 0.1
    test.sourcefun = HEDAC_basic.difsource
    # logsource, difsource, difsquaredsource, divsource, fullcoveragecource generate_difpowersource(0.5) generate_divpowersource(power=2.0)

    test.outputStep = test.T.shape[0]-1 # Output the results at the last time step

    # Normalize the image to create a probability distribution
    prob_dist = image / np.sum(image)

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
    initial = cp.generate_random_point_in_problem().coords.xy
    agent_list = [
        Agent.config(id, (initial[0][0] + random.uniform(-10, 10), initial[1][0] + random.uniform(-10, 10)), capacity=capacity, max_velocity=10)
        for id in range(n_agents)
    ]
    exp = Experiment.Runner(coverage_problem=cp, agents=agent_list)

    exp.solve(profiling_enabled=False)
    # Save the results in a csv file
    (
        totalRouteLength,
        sumOfTaskLengths,
        totalRouteCosts,
        iterations,
        computeTime,
        route_list,
        maxRouteCost,
    ) = exp.evaluateSolution()

    print("file_name",
            totalRouteLength,
            sumOfTaskLengths,
            totalRouteCosts,
            maxRouteCost,
            iterations,
            computeTime,
            cp.getNumberOfTasks(),
            len(agent_list))
    return exp



if __name__ == '__main__':
    # Create a new environment
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    base_path = "data/DemaScenarios/"
    environment_file = "FlatTerrainNature"
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
    ).set_buffer(100).build()

    # Compute the trajectories inside the environment
    paths = compute_trajectories(env, experiment_dir=experiment_dir)

    # convert all the coordinates in the paths to real world coordinates
    world_coordinates = np.zeros_like(paths)
    for i in range(len(paths)):
        agent_path = []
        for j in range(len(paths[i])):
            agent_path.append(env.image_to_world(paths[i][j][0], paths[i][j][1]))
        world_coordinates[i] = agent_path
    
    tasks = [Task.TrajectoryTask(i, LineString(world_coordinates[i]), reward=1) for i in range(len(paths))]
    exp_results = task_allocation(env.polygon.geometry, tasks, n_agents=4, capacity=2000)
    
    # Plot the routes
    plt.figure(figsize=(10, 10))
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
    
    # env.polygon.plot()
    plt.title('Agent Routes')
    plt.axis('equal')
    plt.xlabel('X Coordinate [m]')
    plt.ylabel('Y Coordinate [m]')
    plt.grid(True)
    plt.savefig(experiment_dir + "task_allocation.png")
    plt.show()

    # create a collection of tasks and pickl them to a file
    task_collection = [Task.TrajectoryTask(i, LineString(world_coordinates[i]), reward=1) for i in range(len(paths))]
    
    # TODO oct 17
    # Identify features for all the environments and define some sigma and alpha value to the scenarios, and document it as being static throughout the experiments
    # Generate trajectories using HEDAC
    # Identify how long the trajectories should be when having 30 tasks generated in hedac
    # Document this in the paper as a initial environment definition

    # Create a dataset of tasks and target distribution and save them to a file
    # Remove the extra reward for optimizing the return, this will create better paths for the agents
    
    # Dataset:
    # Create a dataset class, and export it as a pickle:
    # Routes/tasks --> TrajectoryTask in world coordinates, saved as a pickle
    # Heatmap --> numpy

    # TODO Find a way of generating a dataset of tasks 
    # Generate 40 different sets of 30 tasks
    # Save the "target coverage" as a so that it can be used for the evaluation of the allocation
    # save the configuration of the hedac algorithm 
    
    # Environment features: 
    # FlatNature: Lakes, river, wetlands, roads, Forrest edges, 
    # HillyNature: (Possibility of extracting the heightmap?), Forrest edges 
    # Urban: Parks, roads, pathways, Lakes, river, wetlands
    # Water: Wetlands, banks, roads
    
    