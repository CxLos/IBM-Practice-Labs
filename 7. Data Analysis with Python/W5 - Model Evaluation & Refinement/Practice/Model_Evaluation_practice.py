# Import
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Data
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W5 - Model Evaluation & Refinement\Data\auto5.csv'

df = pd.read_csv(path)

# Get only the numeric data:
df = df._get_numeric_data()

# print(df.head(10))
# print(df.dtypes)

# Plotting Functions

# Distribution Plot
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

# Polynomial Plot
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.close()

# PT. 1 TRAINING & TESTING -----------------------------------------------------------

y_data = df['price']
x_data = df.drop('price', axis=1)

# Split data into training and testing data using train_test_split(input_data_x, target_variable_y, test_size_percentage, random_state 1= seed random number generator to ensure the split is the same every time)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# print("number of test samples :", x_test.shape[0])
# print("number of training samples:",x_train.shape[0])

# Simple Linear Regression Object:

# Define Regression Obj
lre = LinearRegression()

# Fit the model using "horsepower"
lre.fit(x_train[['horsepower']], y_train)

# Calculate R^2:
# Test Data
# print('R^2 Test Data:',lre.score(x_test[['horsepower']], y_test))
# Training Data
# print('R^2 Taining Data', lre.score(x_train[['horsepower']], y_train))

# Cross-Validation Score
# Use this if you do not have sufficient data. averages the R^2 from multiple folds of dataset.

# Create variable using cross_val_score(regressionObj, feature_x, target_data_y, cv = # of folds)
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
# print('Cross-Validation Score:', Rcross)

# Calculate avg and std. deviation of our estimate
# print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# Use negative squared error as a score by setting 'scoring' parameter to 'neg_mean_squared_error'
Rcrosss = -1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
# print('Negative Cross-Validation Score:', Rcrosss)

# Predict the output. One fold for testing, the others for training.
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
# print('Cross-Validation Prediction:',yhat[0:5])

# PT.2 OVERFITTING, UNDERFITTING & MODEL SELECTION ----------------------------------

# Multiple Linear Regression Object.
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# Prediction
# Training Data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# print('Training Data prediction:', yhat_train[0:5])

# Test Data
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# print('Test Data prediction:', yhat_test[0:5])

# Distribution of Training Data:
# Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
# DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

# Distribution of Test Data:
# Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

# Overfitting

# Degree 5 Polynomial Model utilizing 55% of the data.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)

# Fit parameters, then transform to obtain modified version of data
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

# Train the linear regression model
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)

# print("Predicted values:", yhat[0:4])
# print("True values:", y_test[0:4].values)

# Polynomial Plot
# PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

# R^2 of training data
# print(poly.score(x_train_pr, y_train))
# R^2 test data
# Negative R^2 value is a sign of overfitting
# print(poly.score(x_test_pr, y_test))

# See how R^2 changes on test data for different order polynomials and then plot it:
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')  
# plt.show()
# plt.close()

# PT. 3 RIDGE REGRESSION -----------------------------------------------------------

# Transform data to 2 degrees
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

# Create Ridge Regression obj and set regularization(alpha) = 1.
RigeModel = Ridge(alpha=1)

RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)

# print('predicted:', yhat[0:4])
# print('test set :', y_test[0:4].values)

# For loop to find the alpha value that minimizes test error:
# Rsqu_test = []
# Rsqu_train = []
# dummy1 = []
# Alpha = 10 * np.array(range(0,1000))
# pbar = tqdm(Alpha)

# for alpha in pbar:
#     RigeModel = Ridge(alpha=alpha) 
#     RigeModel.fit(x_train_pr, y_train)
#     test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
#     pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

#     Rsqu_test.append(test_score)
#     Rsqu_train.append(train_score)

# Plot value of R^2 for different alphas:
# width = 12
# height = 10
# plt.figure(figsize=(width, height))

# plt.plot(Alpha,Rsqu_test, label='validation data  ')
# plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.legend()
# plt.show()
# plt.close()

# PT. 4 GRID SEARCH -----------------------------------------------------------------

# Create dictionary of parameter values
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

RR=Ridge()

# Create Ridge Grid search obj
Grid1 = GridSearchCV(RR, parameters1,cv=4)

Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# Find best parameter value
BestRR = Grid1.best_estimator_
# print(BestRR)

# Now test our model with with best parameter
BestR2 = BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

# print(BestR2)

# QUESTIONS ------------------------------------------------------------------------

# 1. Use tain_test_split() to split up dataset so that 40% of the data samples will be utilized for testing. Set parameter "random_state" = 0.
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)

# print("number of test samples :", x_test1.shape[0])
# print("number of training samples:",x_train1.shape[0])

# 2. Find the R^2 on the test data using 40% of the dataset for testing.
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
# lre.fit(x_train1[['horsepower']], y_train1)

# print('R^2 score horsepower test data 40%:', lre.score(x_test1[['horsepower']], y_test1))


# 3. Calculate the avg R^2 using 2 folds, then find the avg R^2 for the second fold Utilizing the "horsepower" feature.
# Rcross1 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)

# print('Cross-Validation 2-folds:',Rcross1)
# print('Average Cross-Validation of 2-folds:', Rcross1.mean())

# 4a. Create a "PolynomialFeatures" obj "pr1" of degree = 2.
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)
# pr1 = PolynomialFeatures(degree=2)

# 4b. Transform the training and testing samples for the features "horsepower", "curb-weight", "engine-size" and "highway-mpg" utilizing the "fit_transform" method.
# x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# 4c. How many dimensions does the new feature have? Use "shape" attribute.
# print("New feature dimensions:",x_train_pr1.shape)

# 4d. Create a Linear Regression Model "poly1" and then train the obj using the fit method using the polynomial features.
# poly1 = LinearRegression()
# poly1.fit(x_train_pr1, y_train)

# 4e. Use the predict method to predict an output on the polynomial features, then use the "DistributionPlot()" function to display the distribution of the predicted test output vs. the actual test data.
# yhat1 = poly1.predict(x_test_pr1)

# Title = 'Dist Plot of Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test, yhat1, "Actual Values (Test)", "Predicted Values (Test)", Title)

# 4f. Using previous distribution plot, descirbe the two regions where the predicted prices are less accurate than the actual prices.
# From $5,000 to $15,000 and $30,000 to $40,000

# 5. Perform a Ridge Regression. Calculate the R^2 using the polynomial features, use the training data to train the model and use the test data to test the model with alpha parameter = 10.
# RigeModel = Ridge(alpha=10)
# RigeModel.fit(x_train_pr, y_train)

# print('R^2:', RigeModel.score(x_test_pr, y_test))

# 6. Perform a Grid Search for the alpha parameter and the normalization parameter, then find the best values of the parameters.
parameters2 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]

Grid2 = GridSearchCV(RR, parameters2, cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

best_alpha = Grid2.best_params_['alpha']
best_ridge_model = Ridge(alpha=best_alpha)
best_ridge_model.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

print(best_ridge_model)

# Save to '.csv'
# df.to_csv('auto5.csv')