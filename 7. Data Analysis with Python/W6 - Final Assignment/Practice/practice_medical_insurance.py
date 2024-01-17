from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W6 - Final Assignment\Data\Medical_insurance.csv'

# TASK 1: IMPORT DATASET ---------------------------------------------------------------------------------

# 1-1. Print first 10 rows:
df = pd.read_csv(path)
# print(df.info())

# 1-2. Add headers
headers = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
df.columns = headers

# 1-3. Replace all '?' with 'NaN'
df.replace("?", np.nan, inplace=True)

# TASK 2: DATA WRANGLING --------------------------------------------------------------------------------

# 2-1. Use .info() to identify columns with missing values.
# print(df.info())
# missing_data = df.isnull()
# print(missing_data.head(5))

# 2-2. Replace missing values with mean, for categorical attributes replace missing values with most frequent value, update data types of respective columns, then verify.

# Numerical Values
avg_age = df['age'].astype('float').mean(axis=0)
df['age'].replace(np.nan, avg_age, inplace=True)

avg_bmi = df['bmi'].astype('float').mean(axis=0)
df['bmi'].replace(np.nan, avg_bmi, inplace=True)

avg_children = df['children'].astype('float').mean(axis=0)
df['children'].replace(np.nan, avg_children, inplace=True)

avg_charges = df['charges'].astype('float').mean(axis=0)
df['charges'].replace(np.nan, avg_charges, inplace=True)

# Categorical Values
# print(df['sex'].value_counts().idxmax())
df['sex'].replace(np.nan, '2', inplace=True)

# print(df['smoker'].value_counts().idxmax())
df['smoker'].replace(np.nan, '0', inplace=True)

# print(df['region'].value_counts().idxmax())
df['region'].replace(np.nan, '4', inplace=True)

# 2-3. Update 'charges' column to only 2 decimal places
df['charges'] = df['charges'].round(2)

df[['age', 'smoker']] = df[['age','smoker']].astype('int64')

# TASK 3: EXPLORATORY DATA ANALYSIS ---------------------------------------------------------------------

# 3-1. Implement Regression Plot for 'charges' with respect to 'BMI'
# sns.regplot(x="charges", y="bmi", data=df)
# plt.ylim(0,)
# plt.show()
# plt.close()

# 3-2. Implement the box plot for `charges` with respect to `smoker`.
# sns.boxplot(x="charges", y="smoker", data=df)
# plt.show()
# plt.close()

# 3-3. Print the correlation matrix for the dataset.
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
# print("Correlation-Matrix:", correlation_matrix)

# TASK 4: MODEL DEVELOPMENT -----------------------------------------------------------------------------

# 4-1. Fit a linear regression model that may be used to predict the `charges` value, just by using the `smoker` attribute of the dataset. Print the $ R^2 $ score of this model.
lm = LinearRegression()
x = df[['smoker']]
y = df['charges']

lm.fit(x,y)
yhat = lm.predict(x)
r2 = lm.score(x,y)

# print("Predictions smoker:", yhat[0:5])
# print("R^2 smoker:", r2)

# 4-2. Fit a linear regression model that may be used to predict the `charges` value, just by using all other attributes of the dataset. Print the $ R^2 $ score of this model. You should see an improvement in the performance.
lm = LinearRegression()
z = df.drop('charges', axis=1)
y = df['charges']

lm.fit(z,y)
yhat = lm.predict(z)
r2 = lm.score(z,y)

# print("Predictions:", yhat[0:5])
# print("R^2:", r2)

# 4-3. Create a training pipeline that uses `StandardScaler()`, `PolynomialFeatures()` and `LinearRegression()` to create a model that can predict the `charges` value using all the other attributes of the dataset. There should be even further improvement in the performance.
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe = Pipeline(Input)
z = z.astype(float)
pipe.fit(z,y)
ypipe = pipe.predict(z)

mse = mean_squared_error(y,ypipe)
r22 = r2_score(y, ypipe)

# print("MSE all factors:", ypipe)
# print("R^2 all factors:", r22)

# TASK5: MODEL EVALUATION & REFINEMENT ------------------------------------------------------------------

# 5-1. Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.
y_data = df['charges']
x_data = df.drop('charges', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=1)

# 5-2. Initialize a Ridge regressor that used hyperparameter \alpha = 0.1. Fit the model using training data subset. Print the $ R^2 $ score for the testing data.
RR=Ridge(alpha=0.1)

RR.fit(x_train, y_train)
yhat2 = RR.predict(x_test)
R3 = r2_score(y_test, yhat2)

# print("R^2", R3)

# 5-3. Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model, as above, using the training subset. Print the $ R^2 $ score for the testing subset.
pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR.fit(x_train_pr, y_train)
yhat3 = RR.predict(x_test_pr)
R4 = r2_score(y_test, yhat3)

print("predictions:", yhat3[0:5])
print("Actual results:", y_test[0:5])
print("Polynomial R^2 score:", R4)

# -------------------------------------------------------------------------------------------------------

# print(df.head(10))