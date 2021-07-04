# Machine Learning Projects

## Repo Projects

1. Data Preprocessing - Basics of importing data, filling in empty values with averages, split observations between test and training groups. Feature scaling applied to normalize model.

2. Simple Linear Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is linear - chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression (Outcome is numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

4. Polynomial Regression (Outcome is numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is non-linear - chart is visualized as an exponential line graph either upwards or downwards with various points of observations. Formula is Y = MX + MX**2 + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

5. Logistic Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model since Male will become binary as a result of becoming a dummy variable - same goes for any Categorical variable) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.

6. K-NN (K-Nearest Neighbors) where outcome is Binary - Same as Logistic Regression but its Training & Test Models have a more flexible line of separation between Yes/No. Algorithm will assign prediction based on the data point's "Nearest Neighbor."

7. Support Vector Machine (SVM) where outcome is Binary - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary. Yes/No are defined by equal distances of closest polar opposite data points belonging to Yes or No respectively. Visually this can appear as a straight line between Yes/No and may mimic a Logistic Regression but the nearest opposing data points define the boundary line.

8. Kernel SVM (Outcome is Binary) - Same as normal SVM except its boundary line of separation appears as a non-linear curve for data that do not have a linear Yes/No relationship. Predicts Purchase Rate (Y/N) based on Age and Estimated Salary.

9. Support Vector Regression (SVR) where outcome is Numerical - A hyperplane (or tube shaped object) is fitted to a linear plot where data points within the hyperplane are disregarded as "errors" and act instead as a buffer or range. Values outside of this hyperplane become support vectors and shape the regression line.

10. Decision Tree Regression (Outcome is Numerical) - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a numerical score. As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

11. Random Forest Regression (Outcome is Numerical) - Multiple random samples taken from a Decision Tree Regression and trains the model on averages of the samples.

12. Naive Bayes (Outcome is Binary) - Test values are given a small range (or circular area) where its assignment (Yes/No) will depend on the probability of surrounding data points. The more a given sample area has a Yes/No representation the likelihood (or probability) the predicted result will conform to a Yes/No assignment. Not to be confused with K-NN which just looks at the nearest neigbor where Naive Bayes looks at micro sample sizes like an entire city (multiple neighbors) for example.

13. Decision Tree Classifier (Outcome is Binary) - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a binary score (Y/N). As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

14. Random Forest Classifier (Outcome is Binary) - Multiple random samples taken from a Decision Tree Classifier and trains the model on averages of the samples.

15. K-Means Clustering (Outcome is a Grouping) - Not to be confused with Decision Tree Algorithm where we predict a dependent variable. In K-Means we are trying to find groupings within data which may become a dependent variable of itself. To use K-Means first choose number K of clusters (or groups), Select at random K points -> the centers of each group (not necessarily from the dataset) which the centers are equal distance from each other, Assign each data point to the closest center -> That forms K clusters (or groups), Compute and place the new center of each cluster, Reassign each data point to the closest center, If any reassignment took place, recompute otherwise finished. Use the Elbow Method to visually plot the WCSS, choose the amount of clusters where the rate of decline becomes less gradual (or has short drop in %) compared to the next plotted data point.

16. Hierarchical Clustering (Outcome is a Grouping) - Not to be confused with K-Means Algorithm but can often produce the same results. There are two main approaches to HC which are Agglomerative and Divisive - this example will focus on Agglomerative where it focuses on a single data point building a cluster around it based on the proximity of nearby data points. Several options exist how to structure the creation of clusters: choose nearest data points, choose farthest data points, choose average (or Eucleadean) distance between data points, choose distance from centers (like K-Means).


### TODO:
- Create Jupyter Notebook versions of projects
- Attach screenshots/examples of all charts & visuals



## How to Determine Model Accuracy

90% to 100% = Too Good, it may be overfitted with correlating variables, poor sampling or include Training data.

80% to 90% = Very Good, something of a rarity and needs to be re-confirmed just in case.

70% to 80% = Good, this range is ideal and is in line with industry standards.

60% to 70% = Poor, this is sub-standard and may yield a positive result though its impact is not as great.

Below 60% = Trash, throw it out and go through the Training process again.



## How to Determine Model Effectiveness

1. Look at the Confusion Matrix.

2. Look at the coefficients.

3. Look at Adjusted R-Squared.

4. Look at the CAP and predicted accuracy percent % when the average slope reaches 50%.



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



### (Before ML) Linear Regressions are modeled through Ordinary Least Squares in SAS / Gretl

### (Before ML) Logistic Regressions are modeled through Limited Dependent Variable -> Logit -> Binary in SAS / Gretl
