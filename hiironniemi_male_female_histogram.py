"""
Excersice 4 a
Height and weight histograms
DATA.ML.100
Riia Hiironniemi 150271556
Takes test values of heights and weights and plots histograms for both and
separates male and female data.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('male_female_X_test.txt')
y = np.loadtxt('male_female_y_test.txt')

# Male heights and weights
male_heights = x[y == 0, 0]
male_weights = x[y == 0, 1]

# Female heights and weights
female_heights = x[y == 1, 0]
female_weights = x[y == 1, 1]

# Create height histograms
plt.figure(1)
plt.hist(male_heights, bins=10, range=[80, 220], alpha=0.5, label='Male',
         color='blue')
plt.hist(female_heights, bins=10, range=[80, 220], alpha=0.5, label='Female',
         color='pink')
# Add labels and legend
plt.title('Height')
plt.legend(loc='upper right')

# Create weight histograms
plt.figure(2)
plt.hist(male_weights, bins=10, range=[30, 180], alpha=0.5, label='Male',
         color='blue')
plt.hist(female_weights, bins=10, range=[30, 180], alpha=0.5, label='Female',
         color='pink')
# Add labels and legend
plt.title('Weight')
plt.legend(loc='upper right')

# Show the plot
plt.show()

