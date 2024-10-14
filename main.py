import HEDAC_basic
import environment_modelling
import trajallocpy
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString
from trajallocpy import Agent, CoverageProblem, Experiment, Task, Utility
import random
import matplotlib.pyplot as plt

def compute_trajectories(image, env: environment_modelling.Environment, experiment_dir: str = "experiments/"):
    # # Load the image
    # image_path = 'heatmap.png'
    # image = Image.open(image_path)
    # image = image.convert('L')
    # image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # image = np.array(image, dtype=np.float64)
    image_height, image_width = image.shape[:2]
            
    test = HEDAC_basic.HEDAC_basic()

    test.method = 'hedac'
    test.results_dir = experiment_dir
    test.sigma_m = 1 # Envrionemtal variance, for smoothing environments
    test.sigma_c = 2

    # test.method = 'smc'
    # test.results_dir = 'experiments/smc_full'
    # test.sigma_m = 1
    # test.sigma_c = 2

    test.X = np.arange(image.shape[1])
    test.Y = np.arange(image.shape[0])
    test.T = np.arange(50)

    test.samples = image
    test.alpha = 1.0
    test.beta = 0.5
    test.gamma = 0.1
    test.va = 2 # Step size 
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
    return exp.routes



if __name__ == '__main__':
    # Create a new environment
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    base_path = "data/DemaScenarios/"
    environment_file = "FlatTerrainNature"
    experiment_dir = "experiments/" + environment_file + "/"
    env = environment_modelling.Environment(base_path + environment_file + ".geojson")
    sigma_wetland = get_filter_sigma(10, env)
    sigma_roads = get_filter_sigma(30, env)

    heatmap = gaussian_filter(env.heatmaps["roads"],sigma=sigma_roads) + gaussian_filter(env.heatmaps["wetland"],sigma=sigma_wetland) * 0.3
    heatmap = np.array(heatmap, dtype=np.float64)

    # Compute the trajectories inside the environment
    paths = compute_trajectories(heatmap.T, env, experiment_dir=experiment_dir)

    # convert all the coordinates in the paths to real world coordinates
    world_coordinates = np.zeros_like(paths)

    for i in range(len(paths)):
        for j in range(len(paths[i])):
            world_coordinates[i][j] = env.image_to_world(paths[i][j][0], paths[i][j][1])

    # fourier_coef_per_dim = 30
    # get_fourier_coef(heatmap, num_k_per_dim=fourier_coef_per_dim)
    # Export the environment to a geojson file 
    boundary = env.polygon
    tasks = [Task.TrajectoryTask(i, LineString(world_coordinates[i]), reward=1) for i in range(len(paths))]
    routes = task_allocation(boundary.geometry, tasks, n_agents=4, capacity=2000)

    # Plot the routes
    plt.figure(figsize=(10, 10))
    for route in routes.values():
        x, y = zip(*route)
        plt.plot(x, y)
    
    env.polygon.plot()
    plt.title('Agent Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

    # evaluate the performance of the allocation

    # save the trajectories to a numpy file
    np.save(experiment_dir + "trajectories_local.npy", paths)
    np.save(experiment_dir + "trajectories_world.npy", world_coordinates)
    np.save(experiment_dir + "heatmap.npy", heatmap)
    # create the distributions created by the agents paths
