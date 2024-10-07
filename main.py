import HEDAC_basic
import environment_modelling

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import contextily as ctx

def compute_trajectories(image, env: environment_modelling.Environment):
    # # Load the image
    # image_path = 'heatmap.png'
    # image = Image.open(image_path)
    # image = image.convert('L')
    # image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # image = np.array(image, dtype=np.float64)
    image_height, image_width = image.shape[:2]
            
    test = HEDAC_basic.HEDAC_basic()

    test.method = 'hedac'
    test.results_dir = 'experiments/hedac'
    test.sigma_m = 1 # Envrionemtal variance, for smoothing environments
    test.sigma_c = 2

    # test.method = 'smc'
    # test.results_dir = 'experiments/smc_full'
    # test.sigma_m = 1
    # test.sigma_c = 2

    test.X = np.arange(image.shape[1])
    test.Y = np.arange(image.shape[0])
    test.T = np.arange(10)

    test.samples = image
    test.alpha = 1.0
    test.beta = 0.5
    test.gamma = 0.1
    test.va = 2 # Step size 
    test.sigma_ac = 0.1
    # test.kappa = 0.1
    test.sourcefun = HEDAC_basic.difsource
    # logsource, difsource, difsquaredsource, divsource, fullcoveragecource generate_difpowersource(0.5) generate_divpowersource(power=2.0)

    test.outputStep = 100

    # Normalize the image to create a probability distribution
    prob_dist = image / np.sum(image)

    # Flatten the probability distribution and create a list of coordinates
    flat_prob_dist = prob_dist.flatten()
    coordinates = [(i % image_width, i // image_width) for i in range(image_width * image_height)]

    # Sample agent locations based on the probability distribution
    num_agents = 20
    sampled_indices = np.random.choice(len(flat_prob_dist), size=num_agents, p=flat_prob_dist)
    sampled_coordinates = [(coordinates[i][0], coordinates[i][1]) for i in sampled_indices]
    # Assign the sampled coordinates to the agents
    test.agents = sampled_coordinates

    test.search()
    
    # Export the paths to a dict with agent number and the path
    paths = []
    
    for (xa, ya) in zip(test.XA, test.YA):
        paths.append([env.image_to_world(x, y) for x, y in zip(xa, ya)])

    return paths
    
    

if __name__ == '__main__':
    # Create a new environment
    # polygon_file = "data/DemaScenarios/HillyTerrainNature.geojson"
    # polygon_file = "data/DemaScenarios/Urban.geojson"
    # polygon_file = "data/DemaScenarios/Water.geojson"
    polygon_file = "data/DemaScenarios/FlatTerrainNature.geojson"
    env = environment_modelling.Environment(polygon_file)
    
    filter_width_meters = 10  # Example width in meters
    filter_width_pixels = filter_width_meters / env.meter_per_bin
    
    # Calculate the sigma value for the Gaussian filter
    sigma = filter_width_pixels / (2 * np.sqrt(2 * np.log(2)))
        
    heatmap = gaussian_filter(env.heatmaps["roads"],sigma=sigma) + gaussian_filter(env.heatmaps["wetland"],sigma=sigma) * 0.3
    heatmap = np.array(heatmap, dtype=np.float64)
    
    # # Compute the trajectories inside the environment
    # paths = compute_trajectories(heatmap.T, env)

    
    
    
