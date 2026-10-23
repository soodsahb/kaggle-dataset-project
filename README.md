Most people start Machine Learning with model.fit(). For my Month 3 AI Backend Engineering Capstone, I wanted to build the math under the hood first—and then compare it side-by-side with a production Scikit-Learn pipeline. 🧠💻



I just wrapped up an End-to-End Heart Disease Classification Model on clinical patient data (UCI/Kaggle Heart Disease dataset). Instead of treating ML like a black box, I split the project into two parts: a 100% from-scratch implementation using pure NumPy & Pandas, followed by a production-ready Scikit-Learn Pipeline.



Part 1: What I Built From Scratch (Pure NumPy & Pandas) 

🔹 Zero Data Leakage Architecture: Implemented a custom stratified Train (70%) / Validation (15%) / Test (15%) split before touching any statistics. 

🔹 Manual Feature Engineering: Fit median imputation (for missing clinical values) and Z-score standardization strictly on the Training set and applied those learned parameters to Validation and Test sets, alongside One-Hot Encoding with dummy-trap prevention. 

🔹 Vectorized Logistic Regression: Built the forward and backward passes from first principles—Sigmoid activation, Binary Cross-Entropy (Log Loss) with L2 Regularization (excluding the bias term), and vectorized Batch Gradient Descent using matrix multiplication (X.T @ error). 

🔹 Custom Evaluation Engine: Coded Precision, Recall, F1-Score, Confusion Matrix (TP, TN, FP, FN), and a threshold-independent ROC-AUC calculator using Wilcoxon-Mann-Whitney pairwise ranking. 

🔹 Clinical Decision Threshold Tuning: In medical screening, a False Negative (missing a sick patient) is far more dangerous than a False Positive. By tuning the decision threshold on the validation set down to 0.35, I boosted Recall to 89.41% and F1-Score to 85.39%, cutting missed cases in half compared to the default 0.50 cutoff. 

🔹 Model Interpretability & API Readiness: Extracted learned feature weights to identify top risk drivers and wrapped the preprocessing + inference steps into a single predict_single_patient() function ready for a FastAPI endpoint.



Part 2: The Production Scikit-Learn Pipeline To validate my manual math and structure the project for production deployment, I also built a standalone Scikit-Learn Pipeline using: ✅ ColumnTransformer to automatically route numerical columns (SimpleImputer + StandardScaler) and categorical columns (OneHotEncoder) without data leakage. ✅ End-to-end Pipeline coupling preprocessing directly with LogisticRegression, evaluated via classification_report (analyzing per-class support, macro avg, and weighted avg) and roc_auc_score.

Building the calculus and linear algebra by hand made the Scikit-Learn abstractions 10x clearer—especially when debugging class balance, regularization penalties, and probability thresholds.
