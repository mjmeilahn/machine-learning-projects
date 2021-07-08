
import pandas as pd

# THE PURPOSE IS TO MAKE A FASTER CHOICE BASED ON
# AREAS THAT PRODUCE THE HIGHEST ROI...
# AND IGNORE ONES THAT DO NOT

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Sampling Algorithm - No Libraries
import random
rows = 10000 # HARD CODED
columns = 10 # HARD CODED
ads_selected = []
number_of_rewards_1 = [0] * columns
number_of_rewards_0 = [0] * columns
total_reward = 0

for row in range(0, rows):
    ad = 0
    max_random = 0

    for i in range(0, columns):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = i

    ads_selected.append(ad + 1)
    reward = dataset.values[row, ad]

    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1

    total_reward = total_reward + reward

# Visualize Thompson Sampling
import matplotlib.pyplot as plt
plt.hist(ads_selected)
plt.title('Histogram of Selected Ads')
plt.xlabel('Ads')
plt.ylabel('Number of Times Each Ad Was Selected')
plt.show()