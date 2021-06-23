# Machine Learning Projects

## Repo Projects

1. Data Preprocessing - Basics of importing data, filling in empty values with averages, split observations between test and training groups. Feature scaling applied to normalize model.

2. Simple Linear Regression - Predicts Salary based on Years of Experience.

3. Multiple Linear Regression - Predicts Profit based on R&D Spend, Administration, Marketing Spend and State.

4. Polynomial Regression - Predicts Salary based on a unique ID or Job Title.

5. Logistic Regression - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary.

6. K-NN (K-Nearest Neighbors) - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary. Closest distances to data points become classified (Y/N) under K-NN algorithm.

7. Support Vector Machine (SVM) - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary. Yes/No are defined by equal distances of closest polar opposite data points belonging to Yes or No respectively. Visually this can appear as a straight line between Yes/No and may mimic a Logistic Regression but the nearest opposing data points define the boundary line.

8. Kernel SVM - Same as normal SVM except its boundary line of separation appears as a non-linear curve for data that do not have a linear Yes/No relationship. Predicts Purchase Rate (Y/N) based on Age and Estimated Salary.

9. Support Vector Regression (SVR) - A hyperplane (or tube shaped object) is fitted to a linear plot where data points within the hyperplane are disregarded as "errors" and act instead as a buffer or range. Values outside of this hyperplane become support vectors and shape the regression line.

10. Decision Tree Regression - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a numerical score. As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

11. Random Forest Regression - Multiple random samples taken from a Decision Tree Regression and trains the model on averages of the samples.



# TODO: Create EXTRA Jupyter Notebook versions of projects



# TODO: Attach screenshots/examples of all ML visuals



## How to Determine Model Accuracy

90% to 100% = Too Good, it may be overfitted with correlating variables, poor sampling or include Training data.

80% to 90% = Very Good, something of a rarity and needs to be re-confirmed just in case.

70% to 80% = Good, this range is ideal and is in line with industry standards.

60% to 70% = Poor, this is sub-standard and may yield a positive result though its impact is not as great.

Below 60% = Trash, throw it out and go through the Training process again.



## How to Determine Model Effectiveness

Look at the predicted accuracy percent % when the average slope reaches 50%.



## Model Descriptions

Simple Linear Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is linear - chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

Polynomial Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is non-linear - chart is visualized as an exponential line graph either upwards or downwards with various points of observations. Formula is Y = MX + MX**2 + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

Multiple Linear Regression (Outcome is numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

Logistic Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model since Male will become binary as a result of becoming a dummy variable - same goes for any Categorical variable) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.

K-Nearest Neighbors (K-NN) where outcome is Binary - Same as Logistic Regression but its Training & Test Models have a more flexible line of separation between Yes/No. Depends entirely on the dataset and the correlation between relationships.

Support Vector Machine (SVM) where outcome is Binary - Same as Logistic Regression but its margin between Yes/No are defined by equal distances of closest polar opposite data points belonging to Yes or No respectively. Visually this can appear as a straight line between Yes/No and may mimic a Logistic Regression.

Kernel SVM (Outcome is Binary) - Same as normal SVM except its boundary line of separation appears as a non-linear curve for data that do not have a linear Yes/No relationship.

Support Vector Regression (SVR) where outcome is numerical - A hyperplane (or tube shaped object) is fitted to a linear plot where data points within the hyperplane are disregarded as "errors" and act instead as a buffer or range. Values outside of this hyperplane become support vectors and shape the regression line.

Decision Tree Regression (Outcome is numerical) - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a numerical score. As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

Random Forest Regression (Outcome is numerical) - Multiple random samples taken from a Decision Tree Regression and trains the model on averages of the samples.


## Manual Data Prediction (before Python or ML)

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



### (SAS / Gretl) Linear Regressions are modeled through Ordinary Least Squares

### (SAS / Gretl) Logistic Regressions are modeled through Limited Dependent Variable -> Logit -> Binary
