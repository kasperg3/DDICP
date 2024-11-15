import glob
import math
import os
import pickle

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from trajallocpy import Task
from trajgenpy import Utils

from main import *
from main import ExperimentData


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

def plot_reward_function():


    # Define the functions
    def func1(x):
        return -math.log(x)

    def func2(x):
        return math.exp(-x)

    def func3(x):
        return 0.05**x

    # Generate x values
    x = np.linspace(0.01, 1, 400)  # Avoid log(0) by starting from 0.01

    # Compute y values
    y1 = [func1(val) for val in x]
    y2 = [func2(val) for val in x]
    y3 = [func3(val) for val in x]

    # Plot the functions
    plt.figure(figsize=(5, 5))
    plt.plot(x, y1, label=r'$-\log(x)$')
    plt.plot(x, y2, label=r'$\exp(-x)$')
    plt.plot(x, y3, label=r'$0.05^x$')

    # Add labels and legend
    plt.xlabel(r'$\tau$')
    plt.ylabel('score')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 3)
    plt.xlim(0, 1)

    # Show the plot
    plt.show()
    
    
def plot_convergence():
    # Load all convergence.txt files
    file_pattern = 'data/DemaScenariosTasks/FlatTerrainNature_*/convergence.txt'
    files = glob.glob(file_pattern)

    # Initialize a list to store the data
    data_list = []

    # Read data from each file
    for file in files:
        with open(file, 'r') as f:
            data = np.loadtxt(f.readlines()[1:])
        if data.ndim == 1:
            # Reshape if the data is 1-dimensional (single row)
            data = data.reshape(1, -1)
        data_list.append(data)

    # Convert list to a 3D NumPy array (runs, time_steps, columns)
    data_array = np.array(data_list)

    # Check if the data array has the expected shape
    if data_array.ndim != 3:
        raise ValueError("Data array does not have the expected 3 dimensions")

    # Extract time steps and values of interest
    time_steps = data_array[0, :, 0]  # Assuming all files have the same time steps
    values = data_array[:, :, 4]  # Extract the second column from all runs
    # Plot columns 1, 2, 3, and 4
    # for i in range(1, 5):
    #     plt.plot(time_steps, data_array[:, :, i].mean(axis=0), label=f'Column {i}')

    # Add labels and legend
    plt.xlabel('Time Step')
    plt.ylabel('Max difference in information gain')
    plt.title('Mean Values of Columns 1, 2, 3, and 4')
    plt.legend()

    # Show plot
    plt.show()

    # Compute mean and standard error
    mean_values = np.mean(values, axis=0)
    stderr_values = stats.sem(values, axis=0)

    # Compute confidence intervals (95%)
    confidence_interval = 1.96 * stderr_values

    # Plot mean values
    plt.plot(time_steps, mean_values, label='Mean')

    # Plot confidence intervals
    plt.fill_between(time_steps, mean_values - confidence_interval, mean_values + confidence_interval, color='b', alpha=0.2, label='95% Confidence Interval')

    # Add labels and legend
    plt.xlabel('Time Step')
    plt.ylabel('Max difference in information gain')
    plt.legend()

    # Show plot
    plt.show()


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
    plt.figure(figsize=(5, 5))
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


def plot_comparrative_results(*experiment_files,legends =[]):
    plt.figure(figsize=(5, 5))
    for i, experiment_file in enumerate(experiment_files):
        df = pd.read_pickle(experiment_file)
        global_information_samples = []
        information_time_samples = []
        for row in df.iterrows():
            information_time_samples.append(row[1]["information_time"])
            global_information_samples.append(row[1]["global_information_gain"])
            
        # for k, global_info in enumerate(global_information_samples):
        #      for j, info in enumerate(global_info):
        #         global_info[j] *= 0.10**(information_time_samples[k][j]/1200)


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
    plt.ylabel('Information Gain [\%]')
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
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["itCBBA", "HEDAC", "Static Reward", "HEDAC"]
    agent_counts = [2, 3, 4, 5]

    data = []
    timestamp_for_comparrison = 800
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
            category = categories[i // len(categories)]
            agent_count = agent_counts[i % len(agent_counts)]
            data.append([agent_count, category, information_gain_at])
    df = pd.DataFrame(data, columns=["Agents", "Method", "Information gain"])
    
    plt.figure(figsize=(5, 5))
    sns.set_palette("tab10")
    # sns.violinplot(x="Agents", y="Information gain", hue="Method", data=df)
    # sns.boxplot(x="Agents", y="Information gain", hue="Method", data=df)
    sns.pointplot(x="Agents", y="Information gain", hue="Method", data=df, dodge=True, join=False, ci=95)
    # plt.title("Mean Cumulative Information Gain at "+ str(timestamp_for_comparrison) + " seconds for Different Experiments")
    plt.ylabel("Information gain")
    plt.legend(title="Method")
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("data/plots/boxplot_mean_cumulative_information_gain.png")
    # plt.show()


def plot_discounted_information_gain():

    experiment_files = [
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_2_capacity_4000_20241113_125852/20241113_125852.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241113_131536/20241113_131536.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_133333/20241113_133333.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_5_capacity_1600_20241113_135052/20241113_135052.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["itCBBA", "HEDAC", "Static Reward", "HEDAC"]
    agent_counts = [2, 3, 4, 5]
    capacity = [4000, 2666, 2000, 1600]

    data = []
    timestamp_for_comparrison = 600
    for i, experiment_file in enumerate(experiment_files):
        global_information_samples = []
        information_time_samples = []
        df = pd.read_pickle(experiment_file)
        for row in df.iterrows():
            information_time_samples.append(row[1]["information_time"])
            global_information_samples.append(row[1]["global_information_gain"])
        
        for k, global_info in enumerate(global_information_samples):
             for j, info in enumerate(global_info):
                global_info[j] *= 0.10**(information_time_samples[k][j]/capacity[i % len(capacity)])

               
        cumulative_information_samples = [np.cumsum(global_info) for global_info in global_information_samples]
        max_length = max(len(sample) for sample in cumulative_information_samples)
        padded_cumulative_information_samples = [np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in cumulative_information_samples]
        information_time = np.mean([np.pad(sample, (0, max_length - len(sample)), 'edge') for sample in information_time_samples], axis=0)
        for sample in padded_cumulative_information_samples:
            information_gain_at = sample[np.searchsorted(information_time, timestamp_for_comparrison)]
            category = categories[i // len(categories)]
            agent_count = agent_counts[i % len(agent_counts)]
            data.append([agent_count, category, information_gain_at])
    df = pd.DataFrame(data, columns=["Agents", "Method", "Information gain"])
    
    plt.figure(figsize=(5, 5))
    sns.set_palette("tab10")
    # sns.violinplot(x="Agents", y="Information gain", hue="Method", data=df)
    # sns.boxplot(x="Agents", y="Information gain", hue="Method", data=df)
    sns.pointplot(x="Agents", y="Information gain", hue="Method", data=df, dodge=True, join=False, ci=95)
    # plt.title("Mean Cumulative Information Gain at "+ str(timestamp_for_comparrison) + " seconds for Different Experiments")
    plt.ylabel("Information gain")
    plt.legend(title="Method")
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("data/plots/discounted_information_gain.png")
    # plt.show()

def plot_percent_survivors_detected():
    experiment_files = [
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_2_capacity_4000_20241114_110356/20241114_110356.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241114_112203/20241114_112203.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241114_114114/20241114_114114.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_5_capacity_1600_20241114_120058/20241114_120058.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["itCBBA", "HEDAC", "static reward", "HEDAC"]
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
        print(f'{agent_counts[i % len(agent_counts)]},{categories[i // len(categories)]},{mean_detection_time:.2f},{confidence_interval_detection:.2f}')
        for detection_time in detection_times:
            category = categories[i // len(categories)]
            agent_count = agent_counts[i % len(agent_counts)]
            data.append([agent_count, category, detection_time])
    df = pd.DataFrame(data, columns=["Agents", "Method", "Detection time"])
    
    plt.figure(figsize=(5, 5))
    sns.set_palette("tab10")
    sns.pointplot(x="Agents", y="Detection time", hue="Method", data=df, dodge=True, errorbar=("ci",95))#, linestyle="none")
    # sns.violinplot(x="Agents", y="Detection time", hue="Method", data=df)
    # sns.boxplot(x="Agents", y="Detection time", hue="Method", data=df)
    # plt.title(f"Mean time for detecting the first {n_detections} survivors for Different Experiments")
    plt.ylabel("Time [s]")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig("data/plots/mean_detection_time.png")
    plt.savefig("data/plots/mean_detection_time.svg")
    # plt.show()

def plot_reward_function():
    # Define the function
    def func(x, y):
        return 0.10**x * np.exp(y * 10)

    # Generate x and y values
    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 0.1, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # Plot the contour
    plt.figure(figsize=(5, 5))
    contour = plt.contourf(X, Y, Z, levels=8, cmap='magma')
    # plt.colorbar(contour)
    # plt.title(r'Contour plot of $0.10^x \cdot e^{y \cdot 10}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('off')
    # plt.grid(True)
    plt.savefig("data/plots/reward_function.svg", format='svg', bbox_inches='tight', dpi=1000, pad_inches=0)
    # plt.savefig("data/plots/reward_function.png",bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.show()


def generate_table_values():
    experiment_files = [
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_2_capacity_4000_20241114_110356/20241114_110356.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241114_112203/20241114_112203.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241114_114114/20241114_114114.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_5_capacity_1600_20241114_120058/20241114_120058.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_2_capacity_4000_20241114_070149/20241114_070149.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241114_072400/20241114_072400.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241114_074813/20241114_074813.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_5_capacity_1600_20241114_080740/20241114_080740.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_2_capacity_4000_20241113_140924/20241113_140924.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241113_142608/20241113_142608.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241113_144350/20241113_144350.pkl",
        # "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_5_capacity_1600_20241113_150232/20241113_150232.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_2_capacity_4000_20241109_085823/hedac_agents_2_capacity_4000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl",
        "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_5_capacity_1600_20241108_230402/hedac_agents_5_capacity_1600.pkl"
    ]

    categories = ["itCBBA", "HEDAC", "static reward", "HEDAC"]
    agent_counts = [2, 3, 4, 5]
    capacity = [4000, 2666, 2000, 1600]

    n_detections = [5, 25, 50, 75]
    data = []
    for i, experiment_file in enumerate(experiment_files):
        df = pd.read_pickle(experiment_file)
        
        # Plot the mean number of survivors found over time with confidence interval
        all_survivors_found_time = []
        information_time_samples = []
        global_information_samples = []
        for row in df.iterrows():
            all_survivors_found_time.append(row[1]["survivors_found_time"])
            global_information_samples.append(row[1]["global_information_gain"])
            information_time_samples.append(row[1]["information_time"])

        for k, global_info in enumerate(global_information_samples):
             for j, info in enumerate(global_info):
                global_info[j] *= 0.10**(information_time_samples[k][j]/capacity[i % len(capacity)])
                
        mean_detection_values = []
        confidence_detection_values = []
        for detections in n_detections:
            detection_times = [sorted(times)[:detections] for times in all_survivors_found_time if len(times) >= detections]
            detection_times = [times[-1] for times in detection_times]
            mean_detection_time = np.mean(detection_times)
            std_detection_time = np.std(detection_times)
            confidence_interval_detection = 1.96 * std_detection_time / np.sqrt(len(detection_times))
            mean_detection_values.append(mean_detection_time)
            confidence_detection_values.append(confidence_interval_detection)
            
        data.append([agent_counts[i % len(agent_counts)], n_detections, categories[i // len(categories)], 
                 f"{mean_detection_values[0]:.2f}±{confidence_detection_values[0]:.2f}", 
                 f"{mean_detection_values[1]:.2f}±{confidence_detection_values[1]:.2f}", 
                 f"{mean_detection_values[2]:.2f}±{confidence_detection_values[2]:.2f}", 
                 f"{mean_detection_values[3]:.2f}±{confidence_detection_values[3]:.2f}"])
        print(f'{agent_counts[i % len(agent_counts)]},{categories[i // len(categories)]},{mean_detection_values[0]}±{confidence_detection_values[0]}, {mean_detection_values[1]}±{confidence_detection_values[1]}, {mean_detection_values[2]}±{confidence_detection_values[2]}, {mean_detection_values[3]}±{confidence_detection_values[3]}')
    
    df = pd.DataFrame(data, columns=["Agents", "Survivors", "Method",f"{n_detections[0]}",f"{n_detections[1]}",f"{n_detections[2]}",f"{n_detections[3]}"])
    
    df.to_csv("data/plots/detection_time_table.csv", index=False)


if __name__=="__main__":
    # plot_reward_function()
    # exit(0)
    generate_table_values()
    # plot_information_gathered()
    # plot_percent_survivors_detected()
    # plot_discounted_information_gain()
    # plot_comparrative_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241114_114114/20241114_114114.pkl",
                            #   "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl"
                            #   ,legends=["itCBBA","HEDAC"])


    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241113_120808/20241113_120808.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl")
    # plot_comparrative_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_4_capacity_2000_20241113_083943/20241113_083943.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_4_capacity_2000_20241111_214659/20241111_214659.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_4_capacity_2000_20241111_150838/20241111_150838.pkl",
    #                           "experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_4_capacity_2000_20241108_215950/hedac_agents_4_capacity_2000.pkl"
    #                           ,legends=["Exponential Shaping","Information Reward","Static Reward","HEDAC"])
    # 4 Agent comparrison plot
    
    # plot_information_gain_histogram(dataset_dir)
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_exponential_shaping_agents_3_capacity_2666_20241111_223622/20241111_223622.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_information_reward_agents_3_capacity_2666_20241111_154436/20241111_154436.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_allocation_static_reward_agents_3_capacity_2666_20241111_150055/20241111_150055.pkl")
    # evaluate_results("experiments/FlatTerrainNature/FlatTerrainNature_hedac_agents_3_capacity_2666_20241108_180024/hedac_agents_3_capacity_2666.pkl")