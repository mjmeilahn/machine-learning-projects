
import pandas as pd

# THE PURPOSE IS TO MAKE A FASTER CHOICE BASED ON
# AREAS THAT PRODUCE THE HIGHEST ROI...
# AND IGNORE ONES THAT DO NOT

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Upper Confidence Bound Algorithm - No Libraries
import math
rows = 10000 # HARD CODED
columns = 10 # HARD CODED
ads_selected = []
number_of_selections = [0] * columns
sum_of_rewards = [0] * columns
total_reward = 0

for row in range(0, rows):
    ad = 0
    max_upper_bound = 0

    for i in range(0, columns):
        if (number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(row + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400

        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad + 1) # Remove indexes from plotted X-Ticks
    number_of_selections[ad] += 1
    reward = dataset.values[row, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualize Upper Confidence Bound
import matplotlib.pyplot as plt
plt.hist(ads_selected)
plt.title('Histogram of Selected Ads')
plt.xlabel('Ads')
plt.ylabel('Number of Times Each Ad Was Selected')
plt.show()