# Final Project: Generative AI for Data Science

### Paste the following text in the prompt instructions to give the model the appropriate context for the data.

We have a Used Car Price Analysis dataset with the following columns:

1. Model - object
2. Year - int64
3. Transmission - object
    - Manual
    - Automatic
    - Semi-Auto
4. Mileage - int64
5. FuelType - object
    - Petrol
    - Diesel
    - Hybrid
    - Electric
    - Other
6. Tax - float64
7. MPG - float64
8. EngineSize - float64
9. Price - Int64

## 1. Import Dataset

You can begin by using the Generative AI model to create a python script that can load the dataset to a pandas dataframe. The dataset file already has the headers in the first row.

### Write the prompt to generate the said code and test it in the JupyterLite environment. For verification of appropriate loading, include a step for printing the first 5 values of the loaded dataframe.

Write a Python code that can perform the following tasks:
1. Read the CSV file, located on a given file path, into a pandas data frame, assuming that the first row of the file can be used as the headers for the data.
2. Print the first 5 rows of the dataframe to verify correct loading.

```bash
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/used_car_price_analysis.csv'
file_path = os.path.join(script_dir, data_path)
df = pd.read_csv(file_path)

print(df.head())
```

## 2. Data Preparation

At this stage, it is required to clean up the data. As has been informed to you, the data may have missing values and duplicate entries. 

### Write a prompt that performs the following tasks:

Write a Python code that can perform the following tasks:
1. Identifies the columns with missing values and fills the blank cells with average value of the columns.
2. Identifies and drops the duplicate entries from the data.

```bash
import pandas as pd

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Task 1: Identify columns with missing values and fill them with the average value of the columns
df.fillna(df.mean(), inplace=True)

# Task 2: Identify and drop duplicate entries
df.drop_duplicates(inplace=True)

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_used_car_data.csv', index=False)

print("Data cleaning complete. Missing values filled and duplicates dropped.")
```

## 3. Data Insights & Visualizations

Write prompts that generate codes to prform the following actions:

### 1. Identify the 5 attributes that have the highest correlation with the price parameter.

Prompt:

Write a python code that identifies the top 5 attributes with highest correlation with the target attribute in a data frame.

```bash
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Assuming 'Price' is the target attribute
target_attribute = 'Price'

# Get the correlation values for the target attribute, excluding the target itself
correlations = correlation_matrix[target_attribute].drop(target_attribute)

# Identify the top 5 attributes with the highest correlation with the target attribute
top_5_attributes = correlations.abs().nlargest(5).index

print("Top 5 attributes with the highest correlation with the target attribute:" top_5_attributes)
```

### 2. Count the number of cars under each unique value of fuelType attribute.

Prompt:

Write a python code that counts the number of entries in a dataframe with each unique value of a specific attribute.

```bash
import pandas as pd

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the attribute/column you want to analyze
attribute = 'FuelType'  # Replace 'FuelType' with your specific attribute

# Count the number of entries for each unique value of the specified attribute
value_counts = df[attribute].value_counts()

print(f"Number of entries for each unique value in the '{attribute}' column:")
print(value_counts)
```

### 3. Create a Box plot to determine whether cars with automatic, manual or semi-auto type of transmission have more price outliers. Use the Seaborn library for creating the plot.

Prompt:

Write a python code that creates a Seaborn box plot between a source and target attributes of a dataframe.

```bash
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attribute = 'FuelType'  # Replace with your source attribute
target_attribute = 'Price'     # Replace with your target attribute

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=source_attribute, y=target_attribute, data=df)
plt.title(f'Box Plot of {target_attribute} by {source_attribute}')
plt.xlabel(source_attribute)
plt.ylabel(target_attribute)
plt.show()

```

### 4. Generate the regression plot between mpg parameter and the price to determine the correlation type between the two.

Prompt:

Write a python code that creates a regression plot between a source attribute and a target attribute.

```bash
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attribute = 'Mileage'  # Replace with your source attribute
target_attribute = 'Price'    # Replace with your target attribute

# Create the regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=source_attribute, y=target_attribute, data=df)
plt.title(f'Regression Plot of {target_attribute} vs {source_attribute}')
plt.xlabel(source_attribute)
plt.ylabel(target_attribute)
plt.show()

```

## 4. Model Development & Evaluation

#### 1. Fit a linear regression model to predict the price using the feature mpg. Then calculate the R^2 and MSE values for the model.

Pormpt: 

Write a python code to fit a linear regression model on a target variable on basis of a source variable. Calculate the R^2 values and MSE values for this model.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attribute = 'Mileage'  # Replace with your source attribute
target_attribute = 'Price'    # Replace with your target attribute

# Prepare the data
X = df[[source_attribute]]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```

#### 2. Fit a linear regression model to predict the price using the following set of features:

- year
- mileage
- tax
- mpg
- engineSize.

Calculate the R^2 and MSE values for this model.

Prompt:

Write a python code to fit a linear regression model on target variable 'price' on source variables: year, mileage, tax, mpg and engineSize. Calculate the R^2 values and MSE values for this model.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attributes = ['Year', 'Mileage', 'Tax', 'MPG', 'EngineSize']
target_attribute = 'Price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')

```

#### 3. For the same set of features as in the question above, create a pipeline model object that uses standard scalar, second degree polynomial features and a linear regression model. Calculate the R^2 value and the MSE value for this model.

Prompt:

Write a python code that can perform the following tasks.

1. Create a pipeline with standard scalar, second degree polynomial features and linear regression model.
2. Fit this pipeline for a target variable using a set of source attributes from a dataframe.
3. Evaluate the R^2 and MSE values for the trained model.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attributes = ['Year', 'Mileage', 'Tax', 'MPG', 'EngineSize']
target_attribute = 'Price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and LinearRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```

#### 4. For the same set of features, split the data into training and testing data parts. Assume testing part to be 20%. Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

Prompt:

Write a python code that can perform the following tasks.

1. Assuming that a subset of the attributes of a data frame are source attributes and one of the attributes is a target attribute, split the data into training and testing data assuming the testing data to be 20%.
2. Create and fit a Ridge regression model using the training data, setting the regularization parameter to 0.1.
3. Calculate the MSE and R^2 values for the Ridge Regression model using the testing data.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attributes = ['Year', 'Mileage', 'Tax', 'MPG', 'EngineSize']
target_attribute = 'Price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Ridge regression model with regularization parameter alpha=0.1
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```

#### 5. Perform a second order polynomial transform on both the training data and testing data created for the question above. Create and fit a Ridge regression object using the modified training data, set the regularisation parameter to 0.1, and calculate the R^2 and MSE utilising the modified test data.

Prompt:

Write a python code that can perform the following tasks.

1. Assuming that a subset of the attributes of a data frame are source attributes and one of the attributes is a target attribute, split the data into training and testing data assuming the testing data to be 20%.
2. Apply second degree polynomial scaling to the training and testing data.
3. Create and fit a Ridge regression model using the training data, setting the regularization parameter to 0.1.
4. Calculate the MSE and R^2 values for the Ridge Regression model using the testing data.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attributes = ['Year', 'Mileage', 'Tax', 'MPG', 'EngineSize']
target_attribute = 'Price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=0.1))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```

#### 6. In the question above, perform a Grid Search on ridge regression for a set of values of alpha {0.01, 0.1, 1, 10, 100} with 4-fold cross validation to find the optimum value of alpha to be used for the prediction model.

Prompt:

Write a python code that can perform the following tasks.

1. Assuming that a subset of the attributes of a data frame are source attributes and one of the attributes is a target attribute, split the data into training and testing data assuming the testing data to be 20%.
2. Apply second degree polynomial scaling to the training and testing data.
3. Create and fit a Grid search on Ridge regression with cross validation using the training data, for a set of values of the parameter alpha.
4. Calculate the MSE and R^2 values for the Ridge Regression model using the testing data.

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
# Assuming your dataset is in a CSV file named 'used_car_data.csv'
df = pd.read_csv('used_car_data.csv')

# Specify the source and target attributes
source_attributes = ['Year', 'Mileage', 'Tax', 'MPG', 'EngineSize']
target_attribute = 'Price'

# Prepare the data
X = df[source_attributes]
y = df[target_attribute]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, PolynomialFeatures (degree 2), and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge())
])

# Define the parameter grid for alpha
param_grid = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}

# Create and fit the GridSearchCV with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Best alpha: {grid_search.best_params_["ridge__alpha"]}')
print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
```