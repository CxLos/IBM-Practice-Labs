# Import
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Data
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

file_path = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\7. Data Analysis with Python\W5 - Model Evaluation & Refinement\Data\laptop_model_evaluation.csv'

df = pd.read_csv(file_path)
df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'], axis=1, inplace=True)

print(df.head())

# TASK 1: CROSS-VALIDATION -------------------------------------------------------------------------------

# 1-1. Divide the dataset into x_data and y_data parameters. Here y_data is the "Price" attribute, and x_data has all other attributes in the data set.
y_data = df['Price']
x_data = df.drop('Price', axis=1)

# 1-2. Split the data set into training and testing subests such that you reserve 10% of the data set for testing purposes.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# print("number of test samples :", x_test.shape[0])
# print("number of training samples:",x_train.shape[0])

# 1-3. Create a single variable linear regression model using "CPU_frequency" parameter. Print the R^2 value of this model for the training and testing subsets.
lre = LinearRegression()
lre.fit(x_train[['CPU_frequency']], y_train)

# print('CPU_frequency R^2 Test Data:', lre.score(x_test[['CPU_frequency']], y_test))
# print('CPU_frequency R^2 Taining Data', lre.score(x_train[['CPU_frequency']], y_train))

# 1-4. Run a 4-fold cross validation on the model and print the mean value of R^2 score along with its standard deviation.
CPU_R2 = cross_val_score(lre, x_data[['CPU_frequency']], y_data, cv=4)
# print('Mean:', CPU_R2, 'Std:', CPU_R2.std())


# TASK 2: OVERFITTING ------------------------------------------------------------------------------------

# 2-1. Split the data set into training and testing components again, this time reserving 50% of the data set for testing.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.50, random_state=1)

# print("number of test samples :", x_test.shape[0])
# print("number of training samples:",x_train.shape[0])

# 2-2. To identify the point of overfitting the model on the parameter "CPU_frequency", you'll need to create polynomial features using the single attribute. You need to evaluate the R^2 scores of the model created using different degrees of polynomial features, ranging from 1 to 5. Save this set of values of R^2 score as a list.

# Rsqu_test = []
# order = [1, 2, 3, 4, 5]
# for n in order:
#     pr = PolynomialFeatures(degree=n)
#     x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
#     x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])    
#     lre.fit(x_train_pr, y_train)
#     Rsqu_test.append(lre.score(x_test_pr, y_test))

# 2-3. Plot the values of R^2 scores against the order. Note the point where the score drops.

# plt.plot(order, Rsqu_test)
# plt.xlabel('order')
# plt.ylabel('R^2')
# plt.title('R^2 Using Test Data')
# plt.text(3, 0.75, 'Maximum R^2 ')  
# plt.show()
# plt.close()


# TASK 3: RIDGE REGRESSION -------------------------------------------------------------------------------

# 3-1. Now consider that you have multiple features, i.e. 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU' and 'Category'. Create a polynomial feature model that uses all these parameters with degree=2. Also create the training and testing attribute sets.
pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train[['CPU_frequency','RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']])

x_test_pr = pr.fit_transform(x_test[['CPU_frequency','RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']])


# 3-2. Create a Ridge Regression model and evaluate it using values of the hyperparameter alpha ranging from 0.001 to 1 with increments of 0.001. Create a list of all Ridge Regression R^2 scores for training and testing data. 

Rsqu_test = []
Rsqu_train = []
Alpha = 10 * np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# 3-3. Plot the R^2 values for training and testing sets with respect to the value of alpha
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.ylim(0,1)
plt.legend()
plt.show()
plt.close()


# TASK 4: GRID SEARCH ------------------------------------------------------------------------------------

# 4-1. Using the raw data and the same set of features as used above, use GridSearchCV to identify the value of alpha for which the model performs best. Assume the set of alpha values to be used as {0.0001, 0.001, 0.01, 0.1, 1, 10}
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]

# 4-2. Create a Ridge instance and run Grid Search using a 4 fold cross validation.

RR=Ridge()

Grid1 = GridSearchCV(RR, parameters1,cv=4)

# 4-3. Print the R^2 score for the test data using the estimator that uses the derived optimum value of alpha.
Grid1.fit(x_data[['CPU_frequency','RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']], y_data)

BestRR = Grid1.best_estimator_

BestR2 = BestRR.score(x_test[['CPU_frequency','RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']], y_test)

print('Best R^2 Score:',BestR2)