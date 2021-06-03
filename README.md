# Machine Learning Projects

-- Process of Data Prediction (before ML) --

1. Take a large set of data.
2. Identify the numerical or logical dependent variable you want to predict.
3. Split the sample size 50/50 or 70/30.
4. The larger split is used for training the model - the smaller for testing the model.
5. Train the model through Linear or Logical Regression - see repo projects below.
6. (Gretl) Model window -> Analysis -> Forecasts -> Add To Dataset / Plus Icon -> Name variable as "P-Hat" or [variable]-Hat -> Export Data -> Choose ID, Dependent Variable, newly created P-Hat -> Export CSV
7. (Google Sheets) Open CSV export and sort rows by highest P-Hat value
8. (Google Sheets) Open Cumulative Accuracy Profile (CAP) Template, DUPLICATE and fill-in values from CSV export after sorting highest P-Hat value
9. Select % columns then -> Insert -> Chart -> Simple Line Graph
10. This is the END of Training the Model and viewing its CAP
11. Make a copy of the smaller sample size which was not used in Training the model
12. Remove dependent variable you want to predict from the copy
13. Make a new CSV (or file) with Training and Test data
14. (Gretl) Import Training & Test data, re-create variables and run the model -> Test data will not be counted since we removed its dependent variable in STEP 12 -> Model window -> Analysis -> Forecasts (only the Test Data rows) -> Add To Dataset / Plus Icon -> Name variable as "P-Hat" or [variable]-Hat -> Export Data -> Select ID, Dependent Variable, newly create P-Hat -> Export CSV
15. (Google Sheets) Open CSV export and sort rows by highest P-Hat value
16. (Google Sheets) Open Cumulative Accuracy Profile (CAP) Template, DUPLICATE and fill-in values from CSV export after sorting highest P-Hat value
17. Select % columns then -> Insert -> Chart -> Simple Line Graph
18. This is the END of Testing the Model and viewing its CAP
19. Compare both CAPs
20. Repeat earlier steps as needed if accuracy is below 60% or above 90%, 70s to 80s are ideal and ready for full presentation



-- Repo Projects --

1. Data Preprocessing - Basics of importing data, filling in empty values, split observations between test and training groups. Feature scaling applied to normalize model.

-- Below Regressions are modeled through Ordinary Least Squares --

2. Simple Linear Regression (Outcome is numerical) - Requires at least one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). Chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression (Outcome is numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

-- Below Regressions are modeled through Limited Dependent Variable -> Logit -> Binary --

4. Logical Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.