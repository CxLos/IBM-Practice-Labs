import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(filepath)
# print(df.head())
# print(df.columns)

lm = LinearRegression()
# print(lm)

# X - Predictor/ Independent Variable
# Y - Response/ Dependent Variable
X = df[['highway-mpg']]
# Use single bracket to get 1D array
Y = df['price']

# Fit the linear model
lm.fit(X,Y)

# Prediction
Yhat=lm.predict(X)
# print(Yhat[0:5])   

# Results
# [16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085,
#  20345.17153508]

# Intercept value
# print("Intercept is:", lm.intercept_)
# Intercept: 38,423.305851574

# Coefficient
# print("Coefficient is:",lm.coef_)
# Coefficient: -821.73337832

# MULTIPLE LINEAR REGRESSION

# Multiple predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

# print(lm.intercept_)
# print(lm.coef_)

# REGRESSION PLOT

# Highway-mpg reglpot
# width = 12
# height = 10

# plt.figure(figsize=(width, height))
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.ylim(0,)
# plt.show()

# Peak-rpm regression
# plt.figure(figsize=(width, height))
# sns.regplot(x="peak-rpm", y="price", data=df)
# plt.ylim(0,)
# plt.show()

# RESIDUAL PLOT

# Highway-mpg residual
# width = 12
# height = 10

# plt.figure(figsize=(width, height))
# sns.residplot(x=df['highway-mpg'], y=df['price'])
# plt.show()

# MULTIPLE LINEAR REGRESSION

# Prediction
Y_hat = lm.predict(Z)
# print(Y_hat)

# Distribution Plot
# width = 12
# height = 10

# plt.figure(figsize=(width, height))

# ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')

# plt.show()
# plt.close()

# POLYNOMIAL REGRESSION & PIPELINES

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# Fit Polynomial using polyfit(x, y, deg) function, then using poly1d() to display the polynomial func.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

# print(p)

# Plot the function
PlotPolly(p, x, y, 'highway-mpg')
# print(np.polyfit(x, y, 3))

# Polynomia Features object of degree 2:
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)

# print(pr)
# print(Z.shape)
# print(Z_pr.shape)

# PIPELINE

# Create Pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constrctor
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe = Pipeline(Input)

# Convert & Normalize data
Z = Z.astype(float)
pipe.fit(Z,y)
ypipe = pipe.predict(Z)

# print(ypipe[0:4])

# MSE & R^2
mse6 = mean_squared_error(Y, ypipe)
r26  = r2_score(Y, ypipe)
print('The R-square is: ', r26)
print('MSE:', mse6)

# MEASURE FOR IN SAMPLE EVALUATION

# Simple Linear Regression
# highway_mpg_fit
X = df[['highway-mpg']]
Y = df[['price']]
lm.fit(X, Y)
# Find the R^2
# print('The R-square is: ', lm.score(X, Y))

# Calculate Mean Squared Error (MSE)
Yhat=lm.predict(X)
# print('The output of the first four predicted value is: ', Yhat[0:4])

# Compare predicted results with actual results
# mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
mse = mean_squared_error(df['price'], Yhat)
# print('The mean square error of price and predicted value is: ', mse)

# Multiple Linear Regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

# Find the R^2
# print('The R-square is: ', lm.score(Z, df['price']))

# Make prediction
Y_predict_multifit = lm.predict(Z)

# MSE
# print('The mean square error of price and predicted value using multifit is: ', \
#       mean_squared_error(df['price'], Y_predict_multifit))

# Polynomial Fit

# Calculate R^2
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
r_squared = r2_score(y, p(x))
# print('The R-square value is: ', r_squared)

# Calculate MSE
mse = mean_squared_error(df['price'], p(x))
# print(mse)

# PREDICTION & DECISION MAKING

# Create new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat=lm.predict(new_input)
# print(yhat[0:5])

# Plot
# plt.plot(new_input, yhat)
# plt.show()
# plt.close

# QUESTIONS

# 1a. Create a linear Regression Object called "lm1"
lm1= LinearRegression()

# 1b. Train the model using "engine-size" as the independent variable and "price" as the dependent variable.
X = df[['engine-size']]
Y = df[['price']]

lm1.fit(X,Y)

# 1c. Find the slope and Intercept of the model.
# print("Intercept:",lm1.intercept_)
# print("Slope:",lm1.coef_)

# 1d. What is the equation of the predicted line?
# y = b0 + b1 x
yhat = -7963.34 + (166.86*X)
price = -7963.34 + 166.86*df['engine-size']

# print("yhat:", yhat[0:5])
# print("price:", price[0:5])

# 2a. Create and train a Multiple Linear Regression model "lm2" where the response variables is "price", and the predictor variable is "normalized-losses" and "highway-mpg"
lm2= LinearRegression()

Z = df[['normalized-losses', 'highway-mpg']]
Y = df[['price']]

lm2.fit(Z, df['price'])

# 2b. Find the coefficient of the model
# print("Intercept:",lm2.intercept_)
# print("Slope:",lm2.coef_)

# 3. Given the regression plots above, is "peak-rpm" or "highway-mpg" mpore strongly correlated with "price"? Use .corr() method to verify your answer.
# print(df[['peak-rpm','highway-mpg','price']].corr())

# 4. Create an 11 order polynomial model with the variables x, y from above
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f)

# 5. Create a pipeline that standardizes the data, then produces a prediction using a linear regression model using the features Z and target y.
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Input=[('scale', StandardScaler()), ('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z,y)
ypipe = pipe.predict(Z)

# print(ypipe[0:10])

# print(p)
# PlotPolly(p1,x,y,'Highway-mpg')

# df.to_csv('laptop_model_development.csv')