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

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W6 - Final Assignment\Data\kc_house_data.csv'

df = pd.read_csv(filepath)

# TASK 1: IMPORT DATASET ---------------------------------------------------------------------------------

# 1. Display the data types of each column using the function dtypes. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
# print(df.dtypes)
# print(df.describe())


# TASK 2: DATA WRANGLING --------------------------------------------------------------------------------

# 2. Drop the columns "id"  and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe()to obtain a statistical summary of the data. Make sure the inplace parameter is set to True. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
df.drop(['Unnamed: 0','id'], axis=1, inplace=True)
df.dropna(subset=['price'], inplace=True)  
# df.dropna(subset=['price'], inplace=True)  

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

# print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
# print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# print(df.describe())


# TASK 3: EXPLORATORY DATA ANALYSIS ---------------------------------------------------------------------

# 3. Use the method value_counts to count the number of houses with unique floor values, use the method <code>.to_frame()</code> to convert it to a data frame. Take a screenshot of your code and output. You will need to submit the screenshot for the final project. 
# print(df['floors'].value_counts().to_frame())

# 4. Use the function boxplot in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers. Take a screenshot of your code and boxplot. You will need to submit the screenshot for the final project. 
# print(df['waterfront'].value_counts())

# sns.boxplot(x="waterfront", y="price", data=df)
# plt.show()
# plt.close()

# 5. Use the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price. Take a screenshot of your code and scatterplot. You will need to submit the screenshot for the final project. 

# sns.regplot(x="sqft_above", y="price", data=df)
# plt.ylim(0,)
# plt.show()
# plt.close()

# TASK 4: MODEL DEVELOPMENT -----------------------------------------------------------------------------

# 6. Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()

lm.fit(X,Y)
R2 = lm.score(X, Y)
yhat = lm.predict(X)

# print("Sqft_living R^2 score:", R2)

# 7. Fit a linear regression model to predict the <code>'price'</code> using the list of features:
Z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]

Z = Z.astype(float)

# Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
lm.fit(Z,Y)
R2 = lm.score(Z, Y)
yhat2 = lm.predict(Z)

# print("Housing price predictions:", yhat2[0:5])
# print("R^2 score:", R2)

# 8. Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list <code>features</code>, and calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe = Pipeline(Input)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
R2 = r2_score(Y, ypipe)

print("Pipeline predictions:", ypipe[0:5])
print("R^2 score:", R2)

# TASK5: MODEL EVALUATION & REFINEMENT ------------------------------------------------------------------

# 9. Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

RR = Ridge(alpha=0.1)
RR.fit(X, Y)

yhat3 = RR.predict(x_test)
R3 = r2_score(y_test, yhat3)

print(yhat3[0:5])
print(R3)

# 10. Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2. You will need to submit it for the final project.

pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR.fit(x_train_pr, y_train)
yhat4 = RR.predict(x_test_pr)
R4 = r2_score(y_test, yhat4)

print("predictions:", yhat4[0:5])
print("Actual results:", y_test[0:5])
print("Polynomial R^2 score:", R4)

# -------------------------------------------------------------------------------------------------------

# print(df.head())