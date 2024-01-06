"""
Excersice 4 c
Bayes classifier with non-parametric distribution
DATA.ML.100
Riia Hiironniemi 150271556
"""
import numpy as np

# Load training data
X_train = np.loadtxt('male_female_X_train.txt')
y_train = np.loadtxt('male_female_y_train.txt', dtype=int)

# Load test data
X_test = np.loadtxt('male_female_X_test.txt')
y_test = np.loadtxt('male_female_y_test.txt', dtype=int)

# Define the number of bins for height and weight histograms
num_bins = 10
height_range = [80, 220]
weight_range = [30, 180]

# Calculate prior probabilities
prior_male = np.mean(y_train == 0)
prior_female = np.mean(y_train == 1)

# Initialize likelihoods
likelihood_height_male = np.zeros(X_test.shape[0])
likelihood_weight_male = np.zeros(X_test.shape[0])
likelihood_height_female = np.zeros(X_test.shape[0])
likelihood_weight_female = np.zeros(X_test.shape[0])

# Calculate class likelihoods for each test sample
for i in range(X_test.shape[0]):
    height_bin = np.digitize(X_test[i, 0], bins=np.linspace(height_range[0],
                                                            height_range[1],
                                                            num_bins + 1))
    weight_bin = np.digitize(X_test[i, 1], bins=np.linspace(weight_range[0],
                                                            weight_range[1],
                                                            num_bins + 1))

    # Calculate the centroid of the height and weight bins
    height_centroid = (height_bin - 0.5) * (height_range[1] - height_range[0]) / num_bins + height_range[0]
    weight_centroid = (weight_bin - 0.5) * (weight_range[1] - weight_range[0]) / num_bins + weight_range[0]

    # Calculate class likelihoods
    likelihood_height_male[i] = np.histogram(X_train[y_train == 0, 0], bins=np.linspace(height_range[0], height_range[1], num_bins + 1))[0][height_bin - 1]
    likelihood_weight_male[i] = np.histogram(X_train[y_train == 0, 1], bins=np.linspace(weight_range[0], weight_range[1], num_bins + 1))[0][weight_bin - 1]
    likelihood_height_female[i] = np.histogram(X_train[y_train == 1, 0], bins=np.linspace(height_range[0], height_range[1], num_bins + 1))[0][height_bin - 1]
    likelihood_weight_female[i] = np.histogram(X_train[y_train == 1, 1], bins=np.linspace(weight_range[0], weight_range[1], num_bins + 1))[0][weight_bin - 1]

# Calculate the posterior probabilities using Bayes' theorem
posterior_male = prior_male * likelihood_height_male * likelihood_weight_male
posterior_female = prior_female * likelihood_height_female * likelihood_weight_female

# Classify test samples based on the class with higher posterior probability
predictions = np.where(posterior_male > posterior_female, 0, 1)

# Calculate classification accuracy for height only, weight only, and both features
accuracy_height_only = np.mean(predictions == y_test)
accuracy_weight_only = np.mean(likelihood_weight_male > likelihood_weight_female) + np.mean(likelihood_weight_male < likelihood_weight_female)
accuracy_height_weight = accuracy_height_only * accuracy_weight_only

print(f"Accuracy for Height Only: {accuracy_height_only * 100:.2f}%")
print(f"Accuracy for Weight Only: {accuracy_weight_only * 100:.2f}%")
print(f"Accuracy for Height and Weight: {accuracy_height_weight * 100:.2f}%")
