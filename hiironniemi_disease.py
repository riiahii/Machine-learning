"""
Excersice 6
DATA.ML.100
Riia Hiironniemi 150271556
Decision trees
This code uses disease datas for baseline, linear model, decision tree
regression and random forest regression using different sklearn packages.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load training data
x_train = np.loadtxt('disease_X_train.txt')
y_train = np.loadtxt('disease_y_train.txt')
x_test = np.loadtxt('disease_X_test.txt')
y_test = np.loadtxt('disease_y_test.txt')

# a) Baseline
# Calculate the mean
baseline_prediction = np.mean(y_train)

# Print baseline MSE
baseline_mse = mean_squared_error(y_test, np.full_like(y_test,
                                                       baseline_prediction))
print("Baseline Mean Squared Error (MSE):", baseline_mse)

# b) Linear model
# Create a LinearRegression model
linear_model = LinearRegression()

# Fit the linear model to the training data
linear_model.fit(x_train, y_train)

# Make predictions on the test data
predictions = linear_model.predict(x_test)

# Calculate the Mean Squared Error (MSE) for the test set
test_mse = mean_squared_error(y_test, predictions)

# Print the test set MSE
print("Test Set Mean Squared Error (MSE):", test_mse)

# c) Decision tree regressor
# Create a DecisionTreeRegressor model
tree_model = DecisionTreeRegressor(random_state=0)

# Fit the decision tree model to the training data
tree_model.fit(x_train, y_train)

# Make predictions on the test data
predictions = tree_model.predict(x_test)

# Calculate the Mean Squared Error (MSE) for the test set
test_mse = mean_squared_error(y_test, predictions)

# Print the test set MSE
print("Test Set Mean Squared Error (MSE):", test_mse)

# d) Random forest regressor
# Create a RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=0)

# Fit the random forest model to the training data
rf_model.fit(x_train, y_train)

# Make predictions on the test data
predictions = rf_model.predict(x_test)

# Calculate the Mean Squared Error (MSE) for the test set
test_mse = mean_squared_error(y_test, predictions)

# Print the test set MSE
print("Test Set Mean Squared Error (MSE):", test_mse)
