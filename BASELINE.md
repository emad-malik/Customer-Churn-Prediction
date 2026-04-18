# BASELINE.md: Explainable Churn Prediction Pipeline

## 1. Data Acquisition and Initial Cleaning
* Fetch the Kaggle Telco Customer Churn dataset, specifically the expanded snapshot containing 8093 samples and 34 features.
* The target variable is Churn, with a class distribution of roughly 28% positive (Churn) and 72% negative (Non-Churn).
* Locate blank strings in the "Total Charges" column, which occur for brand new customers with zero tenure.
* Trim whitespace from these blank entries, cast them to numeric formats, and explicitly set their values to zero to retain all rows.
* Exclude the "customerID" identifier and any fields that risk leaking future information.

## 2. Preprocessing and Encoding
* Wrap all preprocessing steps inside a single column transformer pipeline so every imputer, encoder, scaler, and selector is fitted only on training folds within the inner cross validation loop.
* Map binary "Yes" and "No" fields to 1 and 0.
* Apply one hot encoding to nominal categorical variables, such as Internet Service, Payment Method, and add on services. 
* Drop a reference level during one hot encoding to avoid perfect multicollinearity for linear and neural models, but you may retain the full set for tree and boosting models.
* Encode the "Contract" variable as ordered integers for tree and boosting models, while using a one hot variant for the linear and multilayer perceptron branches.
* Impute continuous variables (Tenure Months, Monthly Charges, and Total Charges) using the median, apply robust scaling, and consider a log transform for charges if it improves the precision recall area under the curve during the inner search.

## 3. Feature Engineering
* Create "Tenure Bins" to segment customers into 0-6, 7-12, 13-24, 25-36, and above 36 months.
* Create a "High Charge Flag" to mark customers whose Monthly Charges fall above the within fold upper quartile.
* Generate a "Contract by Charge" interaction feature to capture price sensitivity by term.
* Add a "Fiber" flag and a "Fiber by Charge" interaction feature.
* Include the number of technical tickets, cap the count at five, and add a simple quadratic term to model the sharp rise and eventual saturation.
* Create an average revenue proxy called "Charges per Tenure" and run collinearity checks against Monthly Charges.

## 4. Feature Selection Protocol
* Execute a three layer feature selection process entirely within the inner loop.
* **Filter Stage:** Remove constant and quasi constant columns, then retain the strongest candidates using univariate F tests for scaled continuous features and chi square or mutual information for categorical features.
* **Embedded and Wrapper Stage:** Apply an elastic net with a high L1 ratio to zero out redundancies, compute permutation importances with a fast tree ensemble, and run recursive feature elimination using a linear base learner.
* **Stability Selection:** Aggregate the results across the inner folds and keep only the variables that appear in at least three of the five selections.

## 5. Evaluation Protocol: Nested Cross Validation
* Separate model tuning and performance estimation completely by using a nested cross validation strategy.
* Set up the outer loop with 5 fold stratified evaluation to provide unbiased estimates.
* Set up the inner loop by splitting the outer training portion into 3 stratified inner folds for hyperparameter tuning.
* Optimize the hyperparameter search primarily for the area under the precision recall curve, and use ROC AUC as a tiebreaker.
* Select the final decision threshold on the inner validation data to maximize the F1 score, and apply it unchanged to the outer test data.

## 6. Model Configurations and Hyperparameters
Address class imbalance by setting balanced class weights for Logistic Regression, Random Forest, and the Neural Network, and use a positive class scaling term for XGBoost. Tune continuous parameters on a logarithmic scale.

* **Logistic Regression (Elastic Net):** Tune the penalty strength (C) from 1e-4 to 1e4, and the L1 ratio from 0 to 1.
* **Support Vector Classifier (RBF Kernel):** Standardize features, and tune the margin penalty (C) from 1e2 to 1e3 and kernel width (Gamma) from 1e-4 to 1. Calibrate the outputs with a held out inner validation split using Platt scaling or isotonic regression.
* **Random Forest:** Tune the number of estimators from 300 to 500, max depth from 5 to 40, and minimum samples per leaf from 1 to 10.
* **XGBoost:** Tune the number of estimators from 200 to 300, learning rate from 0.01 to 0.30, max depth from 3 to 10, and minimum child weight from 1 to 10. Utilize early stopping on an inner validation slice.
* **Neural Network (Multilayer Perceptron):** Explore hidden layer sizes of (128), (256), or (128, 64) with rectified linear activations. Tune the initial learning rate from 1e-4 to 1e-2 and Alpha (weight decay) from 1e-5 to 1e-2. Use the Adam optimizer, mini batches, and early stopping with patience on validation loss.

## 7. Explainability and Target Results
* Generate SHAP beeswarm plots to visualize global feature importance and direction of effects.
* Create SHAP dependence plots, partial dependence plots (PDP), and individual conditional expectation (ICE) curves specifically for Tenure and Technical Tickets to expose local interactions.
* Ensure the multilayer perceptron hits the target benchmark accuracy of approximately 92.28%, with a precision around 0.88, recall of 0.82, an F1 score of 0.85, ROC AUC of 0.95, and PR AUC of 0.82.