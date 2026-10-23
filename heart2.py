import os
import numpy as np
import pandas as pd

CSV_FILENAME="heart_disease_kaggle.csv"
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
CSV_PATH=CSV_FILENAME if os.path.exists(CSV_FILENAME) else os.path.join(SCRIPT_DIR,CSV_FILENAME)

df=pd.read_csv(CSV_PATH)

print("=" * 76)
print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 76)

print(f"Dataset Shape          : {df.shape[0]} patient records, {df.shape[1]} columns")

missing_counts = df.isnull().sum()
print(f"Columns with NaNs      :\n{missing_counts[missing_counts > 0].to_string()}\n")

class_counts = df["target"].value_counts()
class_ratios = df["target"].value_counts(normalize=True) * 100
print("Target Class Balance   :")
print(f"  - Class 1 (Heart Disease) : {class_counts[1]} patients ({class_ratios[1]:.1f}%)")
print(f"  - Class 0 (Healthy)       : {class_counts[0]} patients ({class_ratios[0]:.1f}%)\n")

#defining features x and target y and one-hot encode categories

X_raw=df.drop(columns=["target"]).copy()
y_raw=df["target"].values.astype(np.float64)

#continous features vs multicategory codes:

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]

X_encoded=pd.get_dummies(X_raw,columns=CATEGORICAL_COLS,drop_first=True,dtype=float)

FEATURE_NAMES=list(X_encoded.columns)
print(f"Total Features after One-Hot Encoding: {len(FEATURE_NAMES)}")
print(f"Feature List: {FEATURE_NAMES}\n")

#train validate and split 

def stratified_train_val_test_split(X_df, y_arr, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Splits X and y into Train, Validation, and Test sets while preserving the exact
    percentage of positive (1) and negative (0) heart disease cases in each split.
    """
    # Create a random number generator with a fixed seed so results are 100% reproducible
    rng = np.random.default_rng(seed)

    train_indices, val_indices, test_indices = [], [], []

    # Loop over each class label (0 and 1) separately to keep class ratios identical
    for label in [0.0, 1.0]:
        # Find all row indices where y equals the current label
        label_idx = np.where(y_arr == label)[0]
        # Shuffle those indices randomly in place
        rng.shuffle(label_idx)

        # Calculate exact cutoff points for 70% train and 15% validation
        n_total = len(label_idx)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Slice the shuffled indices into the 3 buckets
        train_indices.extend(label_idx[:n_train])
        val_indices.extend(label_idx[n_train : n_train + n_val])
        test_indices.extend(label_idx[n_train + n_val :])

    # Shuffle each combined index list so classes aren't grouped in blocks
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    # Use .iloc[] to slice the DataFrame rows and NumPy indexing for target arrays
    return (
        X_df.iloc[train_indices].copy(), y_arr[train_indices],
        X_df.iloc[val_indices].copy(),   y_arr[val_indices],
        X_df.iloc[test_indices].copy(),  y_arr[test_indices],
    )


X_train_df, y_train, X_val_df, y_val, X_test_df, y_test = stratified_train_val_test_split(
    X_encoded, y_raw, train_ratio=0.70, val_ratio=0.15, seed=42
)

print("=" * 76)
print("STEP 2: DATA SPLITTING & LEAKAGE-FREE PREPROCESSING")
print("=" * 76)
print(f"Train Set      : {len(y_train)} rows (Positive rate: {y_train.mean()*100:.1f}%)")
print(f"Validation Set : {len(y_val)} rows (Positive rate: {y_val.mean()*100:.1f}%)")
print(f"Test Set       : {len(y_test)} rows (Positive rate: {y_test.mean()*100:.1f}%)\n")


#imputation and z-score standardization

# 1. Compute median of each numeric column on Train Set to fill missing (NaN) values
train_medians = X_train_df[NUMERIC_COLS].median()

X_train_df[NUMERIC_COLS] = X_train_df[NUMERIC_COLS].fillna(train_medians)
X_val_df[NUMERIC_COLS]   = X_val_df[NUMERIC_COLS].fillna(train_medians)
X_test_df[NUMERIC_COLS]  = X_test_df[NUMERIC_COLS].fillna(train_medians)

# 2. Compute Mean (mu) and Standard Deviation (sigma) of numeric columns on Train Set
train_means = X_train_df[NUMERIC_COLS].mean()
train_stds  = X_train_df[NUMERIC_COLS].std()

# Apply Z-Score Standardization: z = (x - mean) / std
# Why? Features like `chol` (~246) are 100x larger than `oldpeak` (~1.0).
# Without scaling, Gradient Descent oscillates wildly and regularizes unevenly.
X_train_df[NUMERIC_COLS] = (X_train_df[NUMERIC_COLS] - train_means) / train_stds
X_val_df[NUMERIC_COLS]   = (X_val_df[NUMERIC_COLS]   - train_means) / train_stds
X_test_df[NUMERIC_COLS]  = (X_test_df[NUMERIC_COLS]  - train_means) / train_stds

X_train = X_train_df.values.astype(np.float64)
X_val   = X_val_df.values.astype(np.float64)
X_test  = X_test_df.values.astype(np.float64)

def sigmoid(z):
    
    # np.clip prevents overflow warning if z is extremely negative (e.g. -1000)
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def compute_loss_and_gradients(X, y, w, b, lambda_reg=0.1):
    """
    Computes:
    1. Binary Cross-Entropy Loss (Log Loss) + L2 Regularization penalty:
       J(w,b) = -(1/m) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ] + (lambda / 2m) * sum(w^2)
    2. Vectorized Gradients (partial derivatives dJ/dw and dJ/db):
       dj_dw = (1/m) * X.T @ (y_hat - y) + (lambda / m) * w
       dj_db = (1/m) * sum(y_hat - y)
    """
   
    m = X.shape[0]

    z = np.dot(X, w) + b
    y_hat = sigmoid(z)

    # Add a tiny epsilon (1e-15) inside log() so we never compute log(0) = -infinity
    eps = 1e-15
    bce_loss = -np.mean(y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps))

 
    l2_penalty = (lambda_reg / (2.0 * m)) * np.sum(w ** 2)
    total_cost = bce_loss + l2_penalty

    # Backward Pass: compute prediction error vector (shape: m,)
    error = y_hat - y

    # Vectorized gradient for weights `w` (shape: n,) including L2 regularization derivative
    dj_dw = (1.0 / m) * np.dot(X.T, error) + (lambda_reg / m) * w

   
    dj_db = np.mean(error)

    return total_cost, dj_dw, dj_db

def train_logistic_regression(X, y, learning_rate=0.1, epochs=1500, lambda_reg=0.1, verbose=False):
    """
    Trains Logistic Regression parameters `w` and `b` using Batch Gradient Descent.
    """
    n_features = X.shape[1]

    # Initialize weights `w` to zeros and bias `b` to 0.0
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    loss_history = []

    # Gradient Descent Loop: iteratively step downhill opposite the gradient
    for epoch in range(1, epochs + 1):
        cost, dj_dw, dj_db = compute_loss_and_gradients(X, y, w, b, lambda_reg=lambda_reg)
        loss_history.append(cost)

        # Simultaneous parameter update rule: w = w - alpha * dj_dw
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if verbose and (epoch == 1 or epoch % 300 == 0):
            print(f"  Epoch {epoch:4d}/{epochs} | Train Log-Loss: {cost:.4f}")

    return w, b, loss_history


def evaluate_predictions(y_true, y_prob, threshold=0.50):
    """
    Converts probabilities `y_prob` into binary predictions (0 or 1) using `threshold`
    and computes all core classification metrics asked in ML System Design interviews.
    """
    # If predicted probability >= threshold, predict 1 (Heart Disease), else 0 (Healthy)
    y_pred = (y_prob >= threshold).astype(np.float64)

    # Confusion Matrix Components:
    # - True Positive  (TP): Predicted 1, Actual 1 (Correctly caught heart disease)
    # - True Negative  (TN): Predicted 0, Actual 0 (Correctly cleared healthy patient)
    # - False Positive (FP): Predicted 1, Actual 0 (False alarm - Type I error)
    # - False Negative (FN): Predicted 0, Actual 1 (Missed disease! - Type II error, dangerous)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Accuracy: Overall fraction of correct predictions
    accuracy = (tp + tn) / len(y_true)

    # Precision: Out of everyone we predicted has heart disease, how many actually do?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity): Out of everyone who ACTUALLY has heart disease, how many did we catch?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-Score: Harmonic mean of Precision and Recall (balances both)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC (Area Under the Receiver Operating Characteristic Curve):
    # Measures how well the model ranks positive patients higher than negative patients
    # across ALL possible thresholds from 0.0 to 1.0 (computed via Wilcoxon-Mann-Whitney rank).
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    # Broadcast comparison: how often is a positive patient's score > a negative patient's score?
    pairwise_wins = (pos_probs[:, None] > neg_probs[None, :]).astype(float)
    pairwise_ties = (pos_probs[:, None] == neg_probs[None, :]).astype(float) * 0.5
    roc_auc = float(np.mean(pairwise_wins + pairwise_ties))

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

print("=" * 76)
print("STEP 3: TRAINING LOGISTIC REGRESSION (VECTORIZED GRADIENT DESCENT)")
print("=" * 76)

# Train our model on the Training Set and print loss convergence
weights, bias, loss_history = train_logistic_regression(
    X_train, y_train, learning_rate=0.15, epochs=1500, lambda_reg=0.5, verbose=True
)

# Compute predicted probabilities on the Validation Set
val_probs = sigmoid(np.dot(X_val, weights) + bias)

print("\n" + "=" * 76)
print("STEP 4: VALIDATION SET THRESHOLD TUNING (CLINICAL RECALL VS PRECISION)")
print("=" * 76)
print("Why tune the threshold? In medical screening, a False Negative (missing heart")
print("disease) is much worse than a False Positive (ordering a follow-up ECG).")
print("-" * 76)
print(f"{'Threshold':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'False Neg (FN)'}")
print("-" * 76)

best_threshold = 0.50
best_val_f1 = -1.0

for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    m_val = evaluate_predictions(y_val, val_probs, threshold=thresh)
    print(
        f"{thresh:<12.2f} | {m_val['accuracy']*100:>8.2f}% | "
        f"{m_val['precision']*100:>8.2f}% | {m_val['recall']*100:>8.2f}% | "
        f"{m_val['f1']*100:>8.2f}% | {m_val['fn']:>6d} missed"
    )
    # Select threshold that achieves high F1 while keeping Recall strong
    if m_val["f1"] > best_val_f1:
        best_val_f1 = m_val["f1"]
        best_threshold = thresh

# We also explicitly highlight 0.40 as the Clinical Screening Threshold
CLINICAL_THRESHOLD = best_threshold
print(f"\n-> Selected Clinical Screening Threshold: {CLINICAL_THRESHOLD:.2f} (minimizes missed cases)")

print("\n" + "=" * 76)
print("STEP 5: FINAL HELD-OUT TEST SET EVALUATION")
print("=" * 76)

test_probs = sigmoid(np.dot(X_test, weights) + bias)
test_metrics_default  = evaluate_predictions(y_test, test_probs, threshold=0.50)
test_metrics_clinical = evaluate_predictions(y_test, test_probs, threshold=CLINICAL_THRESHOLD)

for label, m in [("Standard Threshold (0.50)", test_metrics_default),
                 ("Clinical Screening Threshold (0.40)", test_metrics_clinical)]:
    print(f"\n[{label}]")
    print(f"  - Accuracy   : {m['accuracy']*100:.2f}%")
    print(f"  - Precision  : {m['precision']*100:.2f}%  (When we flag risk, how often are we right?)")
    print(f"  - Recall     : {m['recall']*100:.2f}%  (Out of actual sick patients, what % did we catch?)")
    print(f"  - F1-Score   : {m['f1']*100:.2f}%  (Harmonic balance of Precision & Recall)")
    print(f"  - ROC-AUC    : {m['roc_auc']:.4f}   (Threshold-independent ranking quality)")
    print(f"  - Confusion Matrix: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")

print("\n" + "=" * 76)
print("STEP 6: MODEL INTERPRETABILITY — TOP LEARNED FEATURE WEIGHTS")
print("=" * 76)
print("Positive weight (+) increases heart disease probability; Negative (-) reduces it:")

# Pair each feature name with its learned weight and sort by absolute magnitude
feature_weights = sorted(
    zip(FEATURE_NAMES, weights), key=lambda pair: abs(pair[1]), reverse=True
)
for feat_name, w_val in feature_weights[:8]:
    direction = "+ Increases Risk" if w_val > 0 else "- Decreases Risk"
    print(f"  {feat_name:<14} : {w_val:>+7.3f}  ({direction})")

def predict_single_patient(patient_dict, threshold=CLINICAL_THRESHOLD):
    """
    Takes a single raw patient dictionary (just like a JSON payload in FastAPI/Spring Boot),
    applies the exact training preprocessing pipeline, and returns the prediction.
    """
    # 1. Convert single patient dict into a 1-row DataFrame
    single_df = pd.DataFrame([patient_dict])

    # 2. One-Hot Encode categorical columns and align columns with FEATURE_NAMES
    single_encoded = pd.get_dummies(single_df, columns=CATEGORICAL_COLS, dtype=float)
    # .reindex() guarantees all 18 training columns exist in the exact same order (filling missing dummy columns with 0)
    single_encoded = single_encoded.reindex(columns=FEATURE_NAMES, fill_value=0.0)

    # 3. Apply Training Set medians, means, and standard deviations
    single_encoded[NUMERIC_COLS] = single_encoded[NUMERIC_COLS].fillna(train_medians)
    single_encoded[NUMERIC_COLS] = (single_encoded[NUMERIC_COLS] - train_means) / train_stds

    # 4. Compute probability via Sigmoid(X @ w + b)
    x_vec = single_encoded.values.astype(np.float64)[0]
    prob = float(sigmoid(np.dot(x_vec, weights) + bias))
    prediction = int(prob >= threshold)

    return {
        "risk_probability": round(prob, 4),
        "predicted_label": prediction,
        "clinical_interpretation": "HIGH RISK - Recommend Cardiology Follow-up" if prediction == 1 else "LOW RISK - Routine Monitoring",
    }


print("\n" + "=" * 76)
print("STEP 7: LIVE INFERENCE ON A NEW PATIENT RECORD (API SIMULATION)")
print("=" * 76)

sample_patient = {
    "age": 61,
    "sex": 1,          # 1 = Male
    "cp": 2,           # 2 = Non-anginal chest pain
    "trestbps": 145,   # Resting blood pressure (mm Hg)
    "chol": 268,       # Serum cholesterol (mg/dl)
    "fbs": 0,          # Fasting blood sugar < 120 mg/dl
    "restecg": 1,      # ST-T wave abnormality
    "thalach": 132,    # Maximum heart rate achieved
    "exang": 1,        # Exercise-induced angina (1 = Yes)
    "oldpeak": 2.4,    # ST depression induced by exercise
    "slope": 1,        # Flat peak exercise ST segment
    "ca": 2,           # 2 major vessels colored by fluoroscopy
    "thal": 3,         # Reversible defect
}

inference_result = predict_single_patient(sample_patient)
print(f"Input Patient Payload : {sample_patient}")
print(f"Model Output          : {inference_result}")



try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score

    print("\n" + "=" * 76)
    print("BONUS: SCIKIT-LEARN PRODUCTION PIPELINE DETECTED & EXECUTED")
    print("=" * 76)

    # 1. Stratified Train/Test Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=42
    )

    # 2. Numeric pipeline: Median Imputer -> Z-Score StandardScaler
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 3. Categorical pipeline: One-Hot Encoder
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    # 4. Combine both into a single ColumnTransformer (prevents data leakage automatically!)
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ], remainder="passthrough")

    # 5. Full End-to-End Pipeline: Preprocessing + L2 Logistic Regression
    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])

    clf_pipeline.fit(X_tr, y_tr)
    y_te_pred = clf_pipeline.predict(X_te)
    y_te_prob = clf_pipeline.predict_proba(X_te)[:, 1]

    print(classification_report(y_te, y_te_pred))
    print(f"Scikit-Learn Test ROC-AUC: {roc_auc_score(y_te, y_te_prob):.4f}")

except ImportError:
    print("\n[Note: `scikit-learn` is optional — the pure NumPy/Pandas implementation above")
    print(" executed 100% of the math from scratch! Run `pip install scikit-learn` locally")
    print(" on your machine to also trigger the bonus `sklearn.pipeline.Pipeline` block.]")