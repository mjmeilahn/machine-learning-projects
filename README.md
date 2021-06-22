# Machine Learning Projects

# TODO: Create EXTRA Jupyter Notebook versions of projects

## Repo Projects

1. Data Preprocessing - Basics of importing data, filling in empty values with averages, split observations between test and training groups. Feature scaling applied to normalize model.

### NOTE: sklearn library handles Backward Elimination process for us

2. Simple Linear Regression - Predicts Salary based on Years of Experience.

3. Multiple Linear Regression - Predicts Profit based on R&D Spend, Administration, Marketing Spend and State.

4. Polynomial Regression - Predicts Salary based on a unique ID or Job Title.

5. Logistic Regression - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary.

6. K-NN (K-Nearest Neighbors) - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary. Closest distances to data points become classified under K-NN algorithm.



## Manual Data Prediction (before ML)

1. Take a large set of data.
2. Identify the numerical or logistic dependent variable you want to predict.
3. Split the sample size 50/50 or 70/30.
4. The larger split is used for training the model - the smaller for testing the model.
5. Train the model through Linear or Logistic Regression - see repo projects below.
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
20. Repeat earlier steps as needed if accuracy is below 60% or above 90%



## How to Determine Model Accuracy

90% to 100% = Too Good, it may be overfitted with correlating variables, poor sampling or include Training data.

80% to 90% = Very Good, something of a rarity and needs to be re-confirmed just in case.

70% to 80% = Good, this range is ideal and as in line with normal findings.

60% to 70% = Poor, this is sub-standard and may yield a positive result though its impact is not as great.

Below 60% = Trash, throw it out and go through the Training process again.



## How to Determine Model Effectiveness

Look at the predicted accuracy percent % when the average slope reaches 50%.



### (Before ML) Below Regressions are modeled through Ordinary Least Squares

Simple Linear Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is linear - chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

Polynomial Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is non-linear - chart is visualized as an exponential line graph either upwards or downwards with various points of observations. Formula is Y = MX + MX**2 + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

Multiple Linear Regression (Outcome is numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

### (Before ML) Logistic Regressions are modeled through Limited Dependent Variable -> Logit -> Binary

Logistic Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model since Male will become binary as a result of becoming a dummy variable - same goes for any Categorical variable) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.

K-NN (K-Nearest Neighbors) where outcome is Binary - Same as Logistic Regression but its Training & Test Models have a more flexible (and accurate) line of separation between Yes/No. Depends entirely on the dataset and the correlation between relationships. A binary prediction should include both Logistic & K-NN.