# Machine Learning Projects

### TODO:
- Remove Hard Coded lines with universal substitutes
- Create Jupyter Notebook versions of projects
- Attach screenshots/examples of all charts & visuals

## Repos

1. Data Preprocessing - Basics of importing data, filling in empty values with averages, split observations between test and training groups. Feature scaling applied to normalize model.

2. Simple Linear Regression (Outcome is Numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is linear - chart is visualized as a linear graph either upwards or downwards with various points of observations. Formula is Y = MX + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

3. Multiple Linear Regression (Outcome is Numerical) - Eliminating insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like Profit). Categorical variables (like State or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model) to avoid the correlation trap. Coefficients are weighted as per unit measure unless the same rate of measure applies across all variables.

4. Polynomial Regression (Outcome is Numerical) - Requires one dependent variable (like Salary) to predict and one independent variable (like Years of Experience) and a constant (starting point like zero). If we assume this data-relationship is non-linear - chart is visualized as an exponential line graph either upwards or downwards with various points of observations. Formula is Y = MX + MX**2 + B. Various tools exist outside of Python to visualize like R, GRETL or a general SAS product. The higher amount of observations lead to a lower P-Value to predict significance.

5. Logistic Regression (Outcome is Binary) - Same method to remove insignificant variables from the model through Backward Elimination (start with many variables) to remove any variable above 0.05 P-Value (re-running model after each removal) to reach high Adjusted R-Squared value. Predicting a lone dependent variable (like whether someone closed an account / binary). Categorical variables (like Region or Gender) are treated as dummies and remove at least one dummy variable from a respective category (like omitting Female dummy variable if we select Male in the model since Male will become binary as a result of becoming a dummy variable - same goes for any Categorical variable) to avoid the correlation trap. Coefficients need to be weighted under E Value calculation i.e. exponent of E. See Google Sheet for example and Heatmap for presentation.

6. K-NN (K-Nearest Neighbors) where Outcome is Binary - Same as Logistic Regression but its Training & Test Models have a more flexible line of separation between Yes/No. Algorithm will assign prediction based on the data point's "Nearest Neighbor."

7. Support Vector Machine (SVM) where Outcome is Binary - Predicts Purchase Rate (Y/N) based on Age and Estimated Salary. Yes/No are defined by equal distances of closest polar opposite data points belonging to Yes or No respectively. Visually this can appear as a straight line between Yes/No and may mimic a Logistic Regression but the nearest opposing data points define the boundary line.

8. Kernel SVM (Outcome is Binary) - Same as normal SVM except its boundary line of separation appears as a non-linear curve for data that do not have a linear Yes/No relationship. Predicts Purchase Rate (Y/N) based on Age and Estimated Salary.

9. Support Vector Regression (SVR) where Outcome is Numerical - A hyperplane (or tube shaped object) is fitted to a linear plot where data points within the hyperplane are disregarded as "errors" and act instead as a buffer or range. Values outside of this hyperplane become support vectors and shape the regression line.

10. Decision Tree Regression (Outcome is Numerical) - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a numerical score. As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

11. Random Forest Regression (Outcome is Numerical) - Multiple random samples taken from a Decision Tree Regression and trains the model on averages of the samples.

12. Naive Bayes (Outcome is Binary) - Test values are given a small range (or circular area) where its assignment (Yes/No) will depend on the probability of surrounding data points. The more a given sample area has a Yes/No representation the likelihood (or probability) the predicted result will conform to a Yes/No assignment. Not to be confused with K-NN which just looks at the nearest neigbor where Naive Bayes looks at micro sample sizes like an entire city (multiple neighbors) for example.

13. Decision Tree Classifier (Outcome is Binary) - Assigns a section to a series of data points that appear as a cluster when visualized. Test model values are passed into the Decision Tree Training algorithm (Y/N) several times until it passes all hurdles to receive a binary score (Y/N). As an example think of how companies weed out job candidates based on select criteria and assign an outcome to each applicant.

14. Random Forest Classifier (Outcome is Binary) - Multiple random samples taken from a Decision Tree Classifier and trains the model on averages of the samples.

15. K-Means Clustering (Outcome is a Grouping) - Not to be confused with Decision Tree Algorithm where we predict a dependent variable. In K-Means we are trying to find groupings within data which may become a dependent variable of itself. To use K-Means first choose number K of clusters (or groups), Select at random K points -> the centers of each group (not necessarily from the dataset) which the centers are equal distance from each other, Assign each data point to the closest center -> That forms K clusters (or groups), Compute and place the new center of each cluster, Reassign each data point to the closest center, If any reassignment took place, recompute otherwise finished. Use the Elbow Method to visually plot the WCSS, choose the amount of clusters where the rate of decline becomes less gradual (or has short drop in %) compared to the next plotted data point.

16. Hierarchical Clustering (Outcome is a Grouping) - Not to be confused with K-Means Algorithm but it can often produce the same results. There are two main approaches to HC which are Agglomerative and Divisive - this example will focus on Agglomerative where it focuses on a single data point building a cluster around it based on the proximity of nearby data points. Several options exist how to structure the creation of clusters: choose nearest data points, choose farthest data points, choose average (or Euclidean) distance between data points, OR choose distance from centers (like K-Means).

17. Apriori (Outcome is Associations) - Algorithm commonly known as "Customers Who Bought X Also Bought Y" where Support, Confidence and Lift are used to measure likelihood of items combined together and the highest Lift determines the winning combination. Example: Movies streamed through Netflix. Support = Watchlists With X / All Watchlists. Confidence = Watchlists With X & Y / Watchlists With X. Lift = Confidence / Support.

18. Eclat (Outcome is Associations) - Same as Apriori except Eclat deals only with Support as its main rule of building associations. In terms of which is better, Apriori has more reliable parameters such as Confidence and Lift. Eclat should be used on a case-by-case basis for associations. Example: Movies streamed through Netflix. Support = Watchlists With X / All Watchlists.

19. Upper Confidence Bound (UCB) where Outcome is Choice - UCB is Deterministic, requires a sample from each variation for a given round and has averages and a boundary to approve or eliminate a variation. It has a fast selection process is based on choices (or conversions) which produce the highest return and will reward the algorithm until it identifies the correct choice. All variations are treated as averages until each begin to move away from a given average range. Those that perform "above the line" are sampled more often until they fall under the average range. Algorithm will repeat until it identifies a variation constantly performing "above the line."

20. Thompson Sampling (Outcome is Choice) - Different than UCB which is Deterministic, requires a sample from each variation for a given round and has averages and a boundary to approve or eliminate a variation. A Thompson Sampling is Probabilistic, does not require a sample from each variation after many rounds as the algorithm will still identify the correct choice. Due to its probabilistic nature and its support of delayed feedback Thompson Sampling is usually the better choice where a sample cannot be provided by each variation per round.

21. Natural Language Processing (Outcome is Binary) - Several types of language processing exist within NLP with few applications in Deep Learning. Bag of Words Model, Chatbot, Speech Translators, Audio Frequency Analysis, Neural Networks, Sequence To Sequence (Seq2Seq). This example will focus on Bag of Words without a Neural Network i.e. Deep Learning application. Few things to keep in mind about this example: given the English language has over 170,000 words, English speakers use roughly 3,000 individual words daily, AND English speakers know roughly 20,000 words total. Therefore our Bag of Words Model will have 20,000 vectors where each word of text will have its own vector. When a given word is used extensively it adds one to the given vector. Over time the data is trained on the values of vectors to respond with Yes/No.

22. Artificial Neural Networks (Outcomes vary from Binary, Numerical and Categorical / Dummy Variables) - One or several Input values are weighted through a Batch (Local, Shorter wait but more inaccurate) or Stochastic (Global, Longer wait but more accurate) Gradient Descent process and its synapses are applied to one or several Neurons in the Hidden Layer. A given Neuron may be trained on one or several Input values which this Neuron has measured to predict the dependent variable. Then an activation function is applied in the Neuron such as Threshold Function, Sigmoid Function, Rectifier and/or Hyperbolic Tangent. Values from the activation function may pass through one or several Hidden Layers with their own activation function or finally be sent to the Output Layer - which has its own function such as Threshold, Sigmoid, Rectifier and/or Hyperbolic Tangent. The output is simply a prediction which is compared to its actual value. A cost function is applied to both predicted vs. actual value and is sent back to the weighted synapses before it passes again to one or several Neurons.

23. Convolutional Neural Networks (Outcome is a Pattern Match) - A given image's pixels are broken into dimensional arrays (2D for Black and White, 3D for Color) where one or several Feature Maps execute an operation on a grid of pixels (rows and columns) one by one to detect a pattern match after one or several filters are applied like edge detection via ReLU (Reticular Linear Units). ReLU assigns an image as Black (negative values) and White (positive values) where Black is re-assigned to a uniform greyscale and the White globs or patches remain acting as an object's edges. Next, Pooling (or Downsampling) is applied which reduces the image and its pixels, rotates and transforms the image into several directions with its own Feature Map to identify matches to a pattern. Then Flattening takes all Feature Maps from the Pooling step and lays out a single column with values from the Pooling Feature Map which will be sent to an Artificial Neural Network as an Input Layer. Essentially the qualifying process to make a final determination on the pattern match is similar to an Artificial Neural Network.

24. Principal Component Analysis (PCA) - TBD