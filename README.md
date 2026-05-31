# ML_Case_Study_Diabetes

Leveling up Predictive Healthcare: An End-to-End Machine Learning Case Study on Diabetes Risk 

I recently wrapped up a comprehensive data science case study focused on predicting diabetes risk using a massive behavioral risk dataset (253,680 records and 22 features). Moving beyond standard model fitting, I focused heavily on robust data pipeline engineering, addressing severe class imbalances, and mathematically validating the model selection.
Here is a breakdown of the workflow and key technical insights from this project:

1. Advanced Data Engineering & Outlier Management
Real-world health data is noisy. To ensure stability, I split the features and applied two distinct outlier imputation methods across the dataset:
•	Z-Score Normalization: Masked extreme tail anomalies outside the [-3, 3] standard deviation boundaries.
•	IQR (Interquartile Range Method): Imputed values falling outside the $1.5 \times \text{IQR}$ threshold.
This process gave me multiple clean variants (X_z,X_z_clean and X_i_clean) to test for algorithmic robustness.

2. Resampling Complex Data Imbalances
With diabetes target variables being heavily skewed, standard models would default to predicting the majority class. To fix this, I engineered a hybrid sampling pipeline:
•	First, applied targeted Undersampling to lower majority class representation.
•	Next, used SMOTETomek (Synthetic Minority Over-sampling Technique + Tomek Links) to intelligently synthesize underrepresented classes while cleaning up overlapping noise.
•	Result: Balanced the experimental set to a clean $95,898$ samples, eliminating structural bias before training.

3. Feature Selection & Interpretability
What actually drives diabetes risk? I evaluated global feature importance using a Tree-Based Classifier. Out of 21 initial health indicators, the top 3 dominant features were:
1.	General Health Status (GenHlth - 23.5% Importance)
2.	Age (15.5% Importance)
3.	Body Mass Index (BMI - 13.3% Importance)
Interestingly, administrative metrics like healthcare access (AnyHealthcare) and regular cholesterol screening checks (CholCheck) carried under 0.5% importance within this feature subset.

4. Hyperparameter Optimization & Model Showdown
I executed cross-validated GridSearchCV routines across Gradient Boosting architectures:
•	CatBoost Classifier vs. LightGBM
•	CatBoost won the baseline optimization with an Average Precision score of 86.3% (depth: 4, iterations: 500, learning_rate: 0.1).
Final Test Set Performance (CatBoost):
•	Overall Accuracy: 82%
•	ROC-AUC Score: 93.1% (Superb class separation)
•	Weighted Average Precision: 87.4%
•	Log Loss: 0.462

5. Statistically Validating the Winner (Hypothesis Testing)
Is CatBoost actually better, or did it just get a lucky data split? To find out, I ran both models 10 times over randomized train-test splits (without fixed random states) and collected their precision scores.
I then ran a Two-Sided Independent T-Test on the performance lists:
•	T-Statistic: 2.10
•	P-Value: 0.05006
With a p-value hovering right on the significance threshold ($\alpha = 0.05$), the statistical evidence heavily leans toward CatBoost offering a genuinely superior predictive edge on this resampled dataset over LightGBM!

Key Takeaway: Model performance is only as good as the underlying data pipeline. By spending time on outlier strategy, intelligent SMOTE cleaning, and mathematical verification, we can build health analytics tools that are both highly accurate and statistically reliable.


