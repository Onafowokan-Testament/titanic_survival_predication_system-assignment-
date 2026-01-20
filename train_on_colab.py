# Google Colab Training Script for Titanic Survival Prediction Model
# Instructions:
# 1. Create a new Google Colab notebook at https://colab.research.google.com/
# 2. Copy and paste this entire script into a single cell
# 3. Run the cell
# 4. Download the model files from the 'Files' panel on the left
# 5. Place them in your local project's model/ folder

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TITANIC SURVIVAL PREDICTION - MODEL TRAINING (Google Colab)")
print("="*70)

# ============================================================================
# 1. LOAD AND EXPLORE THE DATA
# ============================================================================
print("\n[1/7] Loading Titanic Dataset...")
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nStatistical Summary:")
print(df.describe())

# ============================================================================
# 2. DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("[2/7] Data Preprocessing and Feature Engineering...")
print("="*70)

# Select the 5 features as per project requirements
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'

df_working = df[features_to_use + [target]].copy()
print(f"\nWorking dataset shape: {df_working.shape}")
print(f"Missing values before handling:")
print(df_working.isnull().sum())

# Handle missing values - Median imputation
print("\nHandling missing values with median imputation...")
df_working['Age'].fillna(df_working['Age'].median(), inplace=True)
df_working['Fare'].fillna(df_working['Fare'].median(), inplace=True)
df_working.dropna(inplace=True)

print(f"\nMissing values after handling:")
print(df_working.isnull().sum())
print(f"Dataset shape after cleaning: {df_working.shape}")

# Encode categorical variable (Sex)
print("\nEncoding categorical variable (Sex)...")
print("  Male = 1, Female = 0")
df_working['Sex'] = df_working['Sex'].map({'male': 1, 'female': 0})

print(f"\nData after encoding:")
print(df_working.head())

# Prepare features and target
X = df_working[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
y = df_working[target]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())

# ============================================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n" + "="*70)
print("[3/7] Splitting Data into Training and Testing Sets...")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"\nTraining set target distribution:")
print(y_train.value_counts())
print(f"\nTesting set target distribution:")
print(y_test.value_counts())

# Feature scaling
print("\nApplying StandardScaler for feature normalization...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled training data shape: {X_train_scaled.shape}")
print(f"Scaled testing data shape: {X_test_scaled.shape}")

# ============================================================================
# 4. BUILD AND TRAIN MACHINE LEARNING MODEL
# ============================================================================
print("\n" + "="*70)
print("[4/7] Building and Training Logistic Regression Model...")
print("="*70)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n✓ Model trained successfully!")
print(f"\nModel Coefficients:")
print("-" * 50)
for feature, coef in zip(['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'], model.coef_[0]):
    print(f"  {feature:10s}: {coef:8.4f}")
print(f"\n  Intercept: {model.intercept_[0]:8.4f}")

# ============================================================================
# 5. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("[5/7] Evaluating Model Performance...")
print("="*70)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{'ACCURACY METRICS':50s}")
print("-" * 70)
print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n{'CLASSIFICATION REPORT (TEST SET)':50s}")
print("-" * 70)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Did Not Survive', 'Survived']))

print(f"\n{'CONFUSION MATRIX (TEST SET)':50s}")
print("-" * 70)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\n  True Negatives (Correct - Did Not Survive):  {cm[0, 0]}")
print(f"  False Positives (Wrong - Predicted Survived): {cm[0, 1]}")
print(f"  False Negatives (Wrong - Predicted Did Not): {cm[1, 0]}")
print(f"  True Positives (Correct - Survived):         {cm[1, 1]}")

# ============================================================================
# 6. SAVE THE TRAINED MODEL AND SCALER
# ============================================================================
print("\n" + "="*70)
print("[6/7] Saving Model and Scaler...")
print("="*70)

model_path = 'titanic_survival_model.pkl'
scaler_path = 'scaler.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✓ Model saved as: {model_path}")
print(f"✓ Scaler saved as: {scaler_path}")

# ============================================================================
# 7. VERIFY MODEL RELOADING AND PREDICTION
# ============================================================================
print("\n" + "="*70)
print("[7/7] Verifying Model Reloading and Prediction...")
print("="*70)

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

print("\n✓ Models reloaded successfully!")

y_test_pred_reloaded = loaded_model.predict(loaded_scaler.transform(X_test))
reloaded_accuracy = accuracy_score(y_test, y_test_pred_reloaded)

print(f"✓ Accuracy with reloaded model: {reloaded_accuracy:.4f} ({reloaded_accuracy*100:.2f}%)")
print("✓ Model reloading verified successfully!")

# ============================================================================
# 8. TEST PREDICTIONS ON SAMPLE DATA
# ============================================================================
print("\n" + "="*70)
print("[8/8] Testing Predictions on Sample Passengers...")
print("="*70)

new_passengers = pd.DataFrame([
    [1, 0, 35, 1, 512.3292],      # 1st class female, 35
    [3, 1, 25, 0, 7.75],          # 3rd class male, 25
    [2, 1, 40, 2, 21.0],          # 2nd class male, 40
], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])

new_passengers_scaled = loaded_scaler.transform(new_passengers)
predictions = loaded_model.predict(new_passengers_scaled)
probabilities = loaded_model.predict_proba(new_passengers_scaled)

print("\nSample Predictions:")
print("-" * 70)
for i, (_, row) in enumerate(new_passengers.iterrows()):
    survival_status = "SURVIVED" if predictions[i] == 1 else "DID NOT SURVIVE"
    survival_prob = probabilities[i][predictions[i]] * 100
    print(f"\nPassenger {i+1}:")
    print(f"  Class: {int(row['Pclass'])}, Sex: {'Male' if row['Sex']==1 else 'Female'}, "
          f"Age: {row['Age']}, Siblings: {int(row['SibSp'])}, Fare: £{row['Fare']:.2f}")
    print(f"  Prediction: {survival_status} (Confidence: {survival_prob:.2f}%)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"""
✓ Model: Logistic Regression
✓ Features: Pclass, Sex, Age, SibSp, Fare (5 features)
✓ Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
✓ Files created:
  - titanic_survival_model.pkl (The trained model)
  - scaler.pkl (Feature scaler for preprocessing)

NEXT STEPS:
1. Download both .pkl files from the 'Files' panel on the left
2. Place them in your local project's 'model/' folder
3. Run the Streamlit app: streamlit run app.py
4. Test the application locally before deployment

The app will load these pre-trained models and use them for predictions.
No retraining is needed!
""")
