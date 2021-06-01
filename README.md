# Machine Learning Projects

1. Data Preprocessing - Basics of importing data, filling in empty values, split observations between test and training groups. Feature scaling applied to normalize model.

-- Below Regressions are modeled through Ordinary Least Squares --

2. Simple Linear Regression - Requires at least one dependent variable (like Salary) one independent variable (like Years of Experience) and a constant (starting point like zero). Chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value to reach high R-Squared value or Akaike Criterion. Categorical variables (like State) are treated as dummies and remove at least one dummy variable from the model at a time. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.