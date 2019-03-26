# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
file_name = 'progress.csv'
field_to_plot = 'value_loss'
x_axis = 'total_timesteps'
title_of_plot = 'Taxi-v2'

# Load and read data from the progress file
data = pd.read_csv(file_name)

# Plot data
plt.xlabel(x_axis)
plt.ylabel(field_to_plot)
plt.plot(data[x_axis], data[field_to_plot], 'g-', label='PPO') # (x_data, y_data, line_style, legend_label)
plt.title(title_of_plot)
plt.legend(loc='best',  bbox_to_anchor=(0.80, 0.50, 0.5, 0.5))
plt.show()
