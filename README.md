# Machine Learning Projects

1. Data Preprocessing - Basics of importing data, filling in empty values, split observations between test and training groups. Feature scaling applied to normalize model.

-- Below Regressions are modeled through Ordinary Least Squares --

2. Simple Linear Regression (Outcome is numerical) - Requires at least one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). Chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value to reach high Adjusted R-Squared value or Akaike Criterion. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

-- Below Regressions are modeled through Limited Dependent Variable -> Logit -> Binary --

4. Logical Regression (Outcome is Binary)