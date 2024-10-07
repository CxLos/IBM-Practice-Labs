# Heart Disease Dataset

### 1. Paste the following text in the prompt instructions to give the model the appropriate context for the data.

```bash
We have a Heart Disease prediction dataset with a single table which has the following attributes:

1. age - age in years
2. gender- gender (1 = male; 0 = female)
3. cp - chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
5. chol - serum cholestoral in mg/dl
6. fbs - (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
7. restecg - resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or   depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy by Estes criteria
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest
11. slope - the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping
12. ca - number of major vessels (0-3) colored by flourosopy
13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
14. num (the predicted attribute) - diagnosis of heart disease (angiographic disease status)
        -- Value 0: < 50% diameter narrowing
        -- Value 1: > 50% diameter narrowing
```

### 2. Age Distribution

```bash
Write an SQL query to find the minimum, maximum, and average age of patients in the dataset.
```

### 3. Gender Analysis

```bash
Write and SQL query to count the number of male and female patients in the dataset.
```

### 4. Chest Pain Type Frequency

```bash
Write an SQL query to determine the frequency of each type of chest pain (typical angina, atypical angina, non-anginal pain, asymptomatic) among patients."
```

### 5. Age Group Analysis & Target Variable

```bash
Write an SQL query to investigate the distribution of the target variable (presence or absence of heart disease) within different age groups (e.g., 20-30, 30-40, etc.).
```

### 6. Cholesterol Range:

```bash
Find the range of cholesterol levels among patients (minimum, maximum).
```

### 7. Age Range and Gender Analysis:

```bash
Write an SQL query to Determine the age range (youngest and oldest) for male and female patients separately.
```

### 8. Age Group Analysis and Target Variable:

```bash
Write an SQL query to Investigate the distribution of the target variable (presence or absence of heart disease) within different age groups (e.g., 20-30, 30-40, etc.).
```

### 9. Maximum Heart Rate by Age Group:

```bash
Write an SQL query to Find the maximum heart rate achieved during exercise for different age groups (e.g., 30-40, 40-50, etc.).
```

### 10. Percentage of Patients with High Blood Sugar:

```bash
Write an SQL query to Calculate the percentage of patients with fasting blood sugar greater than 120 mg/dl.
```

### 11. Ratio of Patients with Resting Electrocardiographic Abnormality:

```bash
Write an SQL query to Find the ratio of patients with abnormal resting electrocardiographic results to those with normal results.
```

### 12. Number of Patients with Reversible Thalassemia:

```bash
Write an SQL query to Count the number of patients with reversible thalassemia detected by thallium stress testing.
```
### 13. Average Age of Patients with Chest Pain:

```bash
Write an SQL query to Calculate the average age of patients who experienced chest pain during diagnosis.
```
### 14. Distribution of Patients by Number of Major Vessels:

```bash
Write an SQL query to Investigate the distribution of patients based on the number of major vessels colored by fluoroscopy (0-3).
```