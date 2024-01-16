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

# filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W4 - Model Development\Data\laptop_model_development.csv'

df = pd.read_csv(file_path)
# print(df.head())
# print(df.dtypes)

# 1-1. Create single-feature Linear Regression model that fits the pair of "CPU_frequency" and "Price" to find the model for prediction
X = df[['CPU_frequency']]
Y = df['Price']

lm = LinearRegression()
lm.fit(X, Y)

Yhat = lm.predict(X)

# 1-2. Generate Distribution plot for predicted values and that of the actual values. How well do they perform?
# width = 12
# height = 10

# plt.figure(figsize=(width, height))

# ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')

# plt.show()
# plt.close()

# 1-3. Evaluate MSE & R^2 score values
mse = mean_squared_error(df['Price'], Yhat)
r2  = lm.score(X, Y)

# print('The R-square is: ', r2)
# print('MSE:', mse)

# 2-1. Create Multiple Linear Regression using 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', and 'Category'
lm1 = LinearRegression()

Z = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
Y = df['Price']

lm1.fit(Z, Y)
Yhat1 = lm1.predict(Z)

# 2-2. Plot the distribution Graph of the predicted values as well as actual values
# width = 12
# height = 10

# plt.figure(figsize=(width, height))

# ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Yhat1, hist=False, color="b", label="Fitted Values" , ax=ax1)


# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')

# plt.show()
# plt.close()

# 2-3. Find R^2 & MSE value for this fit.
mse1 = mean_squared_error(Y, Yhat1)
r21  = lm1.score(Z, Y)

# print('The R-square is: ', r21)
# print('MSE:', mse1)

# 3-1. Use 'CPU_frequency' to create Polynomial features. Try this for 3 different values of polynomial degrees.

X = X.to_numpy().flatten()

f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)

# 3-2. Plot the regression output against the actual data points to note how the data fits in each case.
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')

    # plt.show()
    # plt.close()

# 3-3. Call the function for 3 models created and get required graphs
PlotPolly(p1, X, Y, 'CPU_frequency')
PlotPolly(p3, X, Y, 'CPU_frequency')
PlotPolly(p5, X, Y, 'CPU_frequency')
# print(np.polyfit(X, Y, 3))

# 3-4. Calculate MSE & R^2
mse2 = mean_squared_error(Y, p1(X))
r22  = r2_score(Y, p1(X))
# print('The R-square is: ', r22)
# print('MSE:', mse2)

mse3 = mean_squared_error(Y, p3(X))
r23  = r2_score(Y, p3(X))
# print('The R-square is: ', r23)
# print('MSE:', mse3)

mse5 = mean_squared_error(Y, p5(X))
r25  = r2_score(Y, p5(X))
# print('The R-square is: ', r25)
# print('MSE:', mse5)

# 4-1. Create a pipeline that performs parameter scaling, Polynomial Feature Generation and Linear Regression. Use the set of multiple features as before to create this pipeline.
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe = Pipeline(Input)

Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)

# 4-2. Calculate MSE and R^2 values
mse6 = mean_squared_error(Y, ypipe)
r26  = r2_score(Y, ypipe)
print('The R-square pipe is: ', r26)
print('MSE pipe:', mse6)

# df.to_csv('laptop_model_development.csv')