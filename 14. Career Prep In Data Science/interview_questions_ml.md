# Machine Learning Interview Questions

### You have a model that has 80% accuracy on the training set, but only 20% accuracy on the test set. What do you think could be causing the problem?

- There is an overfitting problem. Possible reasons could be bad hyper parameters. You should adjust proportion of train/test split. You could also assign a seperate set of the dataset for validation.

### What are the dangers of underfitting?

- 

### What methods would you use to evaluate an imbalanced dataset if false negatives didn't matter?

- I would focus on the metrics that emphasize the positive class primarily with the precision method. This metric measures the accuracy of the positive predictions. It calculates the ratio of true positives and the sum of true positives and false positives. When we have high precision, this indicates that when the model predicts a positive class, it is often correct

### What is the difference between Ridge Regression and Linear Regression?

- Linear Regression: Models relationship between a dependent variable and one or more independent variables by fitting a linear equation. Prone to overfitting especially with complex datasets or when theres too many features. Does not include a penalty term for coefficients which can lead to large coefficient values.

- Ridge Regression includes an L2 regularization penalty term to the loss function. This helps reduce overfitting by discouraging overly complex models. Shrinks coefficients to zero which can lead to a better generalization on unseen data.

Ridge Regression is better when you have multicollinearity or when you want to prevent overfitting your model.

### You have a model that always outputs the number 4. What is the variance?

- 0

### You have a linear system of three equations and two unknowns, How would you, approximately, solve the system?

- Calculate least squares.