Task 3 - Linear Regression
Objective
Implement and understand simple and multiple linear regression using the supplied housing-price dataset.
Dataset
The CSV contains 545 house records and 13 columns. price is the target variable.
What was done
Loaded the CSV using Pandas.
Checked the data types and selected price as the target.
Split the data into 80% training and 20% testing using random_state=42.
Built a Simple Linear Regression model using area to predict price.
Built a Multiple Linear Regression model using all remaining features.
One-hot encoded categorical columns.
Evaluated both models using MAE, RMSE and R².
Created regression and actual-vs-predicted plots.
Saved model coefficients for interpretation.
Results
Simple Linear Regression
MAE: 1,474,748.13
RMSE: 1,917,103.70
R²: 0.2729
Multiple Linear Regression
MAE: 970,043.40
RMSE: 1,324,506.96
R²: 0.6529
The multiple model performs substantially better because house price depends on more than area alone.
Interview Questions - Short Answers
1 What assumptions does linear regression make?
Linear relationship between predictors and target
Independent observations/errors
Constant error variance (homoscedasticity)
Residuals are approximately normal for reliable inference
Low/no problematic multicollinearity among predictors
2 How do you interpret coefficients?
A coefficient is the expected change in predicted target for a one-unit increase in that feature, keeping other features constant. For one-hot encoded categories, it is relative to the reference category.
3 What is R² and its significance?
R² is the proportion of target variance explained by the model. Higher values generally indicate a better fit, but R² alone should not be used to judge a model.
4 When would you prefer MSE over MAE?
MSE penalizes large errors more strongly, so it is useful when large mistakes are especially costly. MAE is easier to interpret and is less sensitive to outliers.
5 How do you detect multicollinearity?
Correlation matrices can provide an initial check. VIF (Variance Inflation Factor) is a common formal diagnostic.
6 Difference between simple and multiple regression?
Simple linear regression uses one predictor. Multiple linear regression uses two or more predictors.
7 Can linear regression be used for classification?
Ordinary linear regression is not the appropriate standard method for classification. Logistic regression or another classification algorithm should be used.
8 What happens if regression assumptions are violated?
Predictions or inference may become unreliable. Depending on the violation, transformations, robust methods, removing problematic variables/outliers, or a different model may help.
