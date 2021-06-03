# Machine Learning Projects

-- Process of General Data Prediction --

1. Take a large set of data.
2. Identify the numerical or logical dependent variable you want to predict.
3. Split the sample size 50/50 or 70/30.
4. The larger split is used for training the model - the smaller for testing the model.

-- Repo Projects --

1. Data Preprocessing - Basics of importing data, filling in empty values, split observations between test and training groups. Feature scaling applied to normalize model.

-- Below Regressions are modeled through Ordinary Least Squares --

2. Simple Linear Regression (Outcome is numerical) - Requires at least one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). Chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression (Outcome is numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

-- Below Regressions are modeled through Limited Dependent Variable -> Logit -> Binary --

4. Logical Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.