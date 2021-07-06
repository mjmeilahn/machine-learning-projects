# Machine Learning Projects

### TODO:
- Create Jupyter Notebook versions of projects
- Attach screenshots/examples of all charts & visuals

## Repos

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

16. Hierarchical Clustering (Outcome is a Grouping) - Not to be confused with K-Means Algorithm but it can often produce the same results. There are two main approaches to HC which are Agglomerative and Divisive - this example will focus on Agglomerative where it focuses on a single data point building a cluster around it based on the proximity of nearby data points. Several options exist how to structure the creation of clusters: choose nearest data points, choose farthest data points, choose average (or Euclidean) distance between data points, OR choose distance from centers (like K-Means).

17. Apriori (Outcome is Associations) - Algorithm also known as "Customers Who Bought X Also Bought Y" where Support, Confidence and Lift are used to measure likelihood of items combined together and the highest Lift determines the winning combination. Example: Movies streamed through Netflix. Support = Watchlists With X / All Watchlists. Confidence = Watchlists With X & Y / Watchlists With X. Lift = Confidence / Support.

18. Eclat (Outcome is Associations) - TBD