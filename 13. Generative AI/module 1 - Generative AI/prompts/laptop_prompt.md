# Laptop Pricing Prompts

### 1. Write a Python code that can perform the following tasks:

```bash
Read the CSV file, located on a given file path, into a Pandas data frame, assuming that the first rows of the file are the headers for the data.
```

### 2. You can now ask the Generative AI model to generate a script to handle the missing data. First, use the model to identify the attributes with missing data. For this, you may run the following query.

```bash
Write a Python code that identifies the columns with missing values in a pandas data frame.
```

### 3. Write a Python code to replace the missing values in a pandas data frame, per the following guidelines.

```bash
- For a categorical attribute "Screen_Size_cm", replace the missing values with the most frequent   value in the column.
- For a continuous value attribute "Weight_kg", replace the missing values with the mean value of the entries in the column.
```

### 4. Further, you should update both attributes’ data type to floating values. You should see a similar response to the following prompt:

```bash
Write a Python code snippet to change the data type of the attributes "Screen_Size_cm" and "Weight_kg" of a data frame to float.
```

### 5. Write a Python code to modify the contents under the following attributes of the data frame as required.

```bash
- Data under 'Screen_Size_cm' is assumed to be in centimeters. Convert this data into inches. Modify the name of the attribute to 'Screen_Size_inch'.
- Data under 'Weight_kg' is assumed to be in kilograms. Convert this data into pounds. Modify the name of the attribute to 'Weight_pounds'.
```

### 6. It may also be required to normalize the data under some attributes. Since there are many normalization forms, mentioning the exact needs and tasks is important. Also, you can save the normalized data as a new attribute or change the original attribute. You need to ensure that all the details of the prompt are clear. For example, let us assume that the data under “CPU_frequency” needs to be normalized w.r.t. the max value under the attribute. You need the changes to be reflected directly under the attribute instead of creating a new attribute.

```bash
Write a Python code to normalize the content under the attribute "CPU_frequency" in a data frame df concerning its maximum value. Make changes to the original data, and do not create a new attribute.
```

### 7. For predictive modeling, the categorical variables are not usable currently. So, you must convert the important categorical variables into indicator numerical variables. Indicator variables are typically new attributes, with content being 1 for the indicated category and 0 for all others. Once you create the indicator variables, you may drop the original attribute. For example, assume that attribute Screen needs to be converted into individual indicator variables for each entry. Once done, the attribute Screen needs to be dropped.

```bash
Write a Python code to perform the following tasks:

1. Convert a data frame df attribute "Screen", into indicator variables, saved as df1, with the naming convention "Screen_<unique value of the attribute>".
2. Append df1 into the original data frame df.
3. Drop the original attribute from the data frame df.
```

### 8. Create a prompt to generate a Python code that converts the values under Price from USD to Euros.

```bash
Write a Python code to convert values under price from 'USD' to 'Euros'
```

### 9. Modify the normalization prompt to perform min-max normalization on the CPU_frequency parameter.

```bash
Write a Python code to perform a min-max normalization on the 'CPU_frequency' parameter
```