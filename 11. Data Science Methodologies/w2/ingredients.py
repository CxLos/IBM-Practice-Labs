
# --------------------------------- FROM UNDERSTANDING TO PREPARATION ----------------------------- #

# ------------------ Imports ------------------ #

import pandas as pd # import library to read data into dataframe
pd.set_option('display.max_columns', None)
import numpy as np # import numpy library
import re # import library for regular expression
import os
import sys

# ------------------ Load Data ------------------ #

current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/ingredients.csv'
file_path = os.path.join(script_dir, data_path)
recipes = pd.read_csv(file_path)
# df = data.copy()

# recipes = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0103EN-SkillsNetwork/labs/Module%202/recipes.csv")

# print("Data read into dataframe!") # takes about 30 seconds

# recipes.head()

# ------------------ Explore Data ------------------ #

# print(recipes.shape) #(57691, 384)
# print(recipes.columns)
# print(recipes.dtypes)

# ---------------------- Data Preparation -----------------------

# Do the ingredients exist in the dataset?
ingredients = list(recipes.columns.values)

# print(

#   # Iterate over each ingredient
#   [match.group(0) for ingredient in ingredients for match in 
#   #  Search for the ingredient in the ingredient list
#    [(re.compile(".*(rice).*"))
#     #  If the ingredient is found, return the ingredient
#     .search(ingredient)] if match])

# print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(wasabi).*")).search(ingredient)] if match])

# print([match.group(0) for ingredient in ingredients for match in [(re.compile(".*(soy).*")).search
# (ingredient)] if match])

# print(recipes["country"].value_counts()) # frequency table

column_names = recipes.columns.values
column_names[0] = "cuisine"
recipes.columns = column_names

# cuisine names to lowercase
recipes["cuisine"] = recipes["cuisine"].str.lower()

# print(recipes['cuisine'].value_counts())

# Replace the cuisine names with the correct names

recipes.loc[recipes["cuisine"] == "austria", "cuisine"] = "austrian"
recipes.loc[recipes["cuisine"] == "belgium", "cuisine"] = "belgian"
recipes.loc[recipes["cuisine"] == "china", "cuisine"] = "chinese"
recipes.loc[recipes["cuisine"] == "canada", "cuisine"] = "canadian"
recipes.loc[recipes["cuisine"] == "netherlands", "cuisine"] = "dutch"
recipes.loc[recipes["cuisine"] == "france", "cuisine"] = "french"
recipes.loc[recipes["cuisine"] == "germany", "cuisine"] = "german"
recipes.loc[recipes["cuisine"] == "india", "cuisine"] = "indian"
recipes.loc[recipes["cuisine"] == "indonesia", "cuisine"] = "indonesian"
recipes.loc[recipes["cuisine"] == "iran", "cuisine"] = "iranian"
recipes.loc[recipes["cuisine"] == "italy", "cuisine"] = "italian"
recipes.loc[recipes["cuisine"] == "japan", "cuisine"] = "japanese"
recipes.loc[recipes["cuisine"] == "israel", "cuisine"] = "israeli"
recipes.loc[recipes["cuisine"] == "korea", "cuisine"] = "korean"
recipes.loc[recipes["cuisine"] == "lebanon", "cuisine"] = "lebanese"
recipes.loc[recipes["cuisine"] == "malaysia", "cuisine"] = "malaysian"
recipes.loc[recipes["cuisine"] == "mexico", "cuisine"] = "mexican"
recipes.loc[recipes["cuisine"] == "pakistan", "cuisine"] = "pakistani"
recipes.loc[recipes["cuisine"] == "philippines", "cuisine"] = "philippine"
recipes.loc[recipes["cuisine"] == "scandinavia", "cuisine"] = "scandinavian"
recipes.loc[recipes["cuisine"] == "spain", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "portugal", "cuisine"] = "spanish_portuguese"
recipes.loc[recipes["cuisine"] == "switzerland", "cuisine"] = "swiss"
recipes.loc[recipes["cuisine"] == "thailand", "cuisine"] = "thai"
recipes.loc[recipes["cuisine"] == "turkey", "cuisine"] = "turkish"
recipes.loc[recipes["cuisine"] == "vietnam", "cuisine"] = "vietnamese"
recipes.loc[recipes["cuisine"] == "uk-and-ireland", "cuisine"] = "uk-and-irish"
recipes.loc[recipes["cuisine"] == "irish", "cuisine"] = "uk-and-irish"

# get the list of cuisines to keep
recipes_counts = recipes["cuisine"].value_counts()
cuisines_indices = recipes_counts > 50

# convert indices to list
cuisines_to_keep = list(np.array(recipes_counts.index.values)[np.array(cuisines_indices)])
# print(cuisines_to_keep)

rows_before = recipes.shape[0] # number of rows of original dataframe
# print("Number of rows of original dataframe is {}.".format(rows_before))

recipes = recipes.loc[recipes['cuisine'].isin(cuisines_to_keep)]

rows_after = recipes.shape[0] # number of rows of processed dataframe
# print("Number of rows of processed dataframe is {}.".format(rows_after))

# print("{} rows removed!".format(rows_before - rows_after))

# convert all Yes's to 1's and the No's to 0's
recipes = recipes.replace(to_replace="Yes", value=1)
recipes = recipes.replace(to_replace="No", value=0)

# check to see which cuisine has all the ingredients below:
check_recipes = recipes.loc[
    (recipes["rice"] == 1) &
    (recipes["soy_sauce"] == 1) &
    (recipes["wasabi"] == 1) &
    (recipes["seaweed"] == 1)
]

# print(check_recipes)

# sum each column
# indexing all rows and all columns starting from the second column
ing = recipes.iloc[:, 1:].sum(axis=0)

# define each column as a pandas series
ingredient = pd.Series(ing.index.values, index = np.arange(len(ing)))
count = pd.Series(list(ing), index = np.arange(len(ing)))

# create the dataframe
ing_df = pd.DataFrame(dict(ingredient = ingredient, count = count))
ing_df = ing_df[["ingredient", "count"]]
# print(ing_df.to_string())

# sort the dataframe in descending order
ing_df.sort_values(["count"], ascending=False, inplace=True)
ing_df.reset_index(inplace=True, drop=True)

# print(ing_df)

# find mean of each ingredient
# cuisines = recipes.groupby("cuisine").mean()
cuisines = recipes.groupby("cuisine").mean()*100

# now convert cuisines to a list:
cuisines_sorted = cuisines.sum(axis=1).sort_values(ascending=False)

# sort cuisines in descending order:
# cuisines_sorted = cuisines.sort_values(ascending=False, inplace=True)
# print(cuisines_sorted)

# print(cuisines.head())

# 
num_ingredients = 4 # define number of top ingredients to print

# define a function that prints the top ingredients for each cuisine
# def print_top_ingredients(row):
#     print(row.name.upper())
#     row_sorted = row.sort_values(ascending=False)*100
#     top_ingredients = list(row_sorted.index.values)[0:num_ingredients]
#     row_sorted = list(row_sorted)[0:num_ingredients]

#     for ind, ingredient in enumerate(top_ingredients):
#         print("%s (%d%%)" % (ingredient, row_sorted[ind]), end=' ')
#     print("\n")

# # apply function to cuisines dataframe
# create_cuisines_profiles = cuisines.apply(print_top_ingredients, axis=1)
# print(create_cuisines_profiles)

# -------------------------------------- Export Database -------------------------------------- #

# updated_path = 'data/ingredients.csv'
# data_path = os.path.join(script_dir, updated_path)
# recipes.to_csv(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# ---------------------------------------------------------------------------------------------