import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle


    
import numpy as np
import glob
import scipy.stats as stats
import matplotlib.pyplot as plt

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
plt.ylabel('Value')
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
plt.ylabel('Value')
plt.title('Confidence Plot of FlatTerrain_Nature_ Runs')
plt.legend()

# Show plot
plt.show()