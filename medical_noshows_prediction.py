# Medical Appointment No-Shows Prediction Mini Project
# Author: Nishant
# Date: August 6, 2025

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Download and Read the Dataset
print("=== Medical Appointment No-Shows Prediction ===")
print("\n1. Reading the Dataset...")

# Note: You'll need to download the dataset from Kaggle first
# https://www.kaggle.com/joniarroba/noshowappointments
# For this example, assuming the file is named 'noshowappointments.csv'

try:
    df = pd.read_csv('noshowappointments.csv')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
except FileNotFoundError:
    print("Dataset file not found. Please download from:")
    print("https://www.kaggle.com/joniarroba/noshowappointments")
    print("and place it in the same directory as this script.")
    exit()

# 2. Check for Missing Values
print("\n2. Checking for Missing Values...")
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

if df.isnull().sum().sum() > 0:
    print("Dropping rows with missing values...")
    df = df.dropna()
    print(f"New dataset shape after dropping missing values: {df.shape}")
else:
    print("No missing values found!")

# 3. Feature Extraction
print("\n3. Feature Extraction...")

# Extract the required features
# Note: Adjusting column names based on typical dataset structure
feature_columns = [
    'Gender', 'Age', 'Scholarship', 'Hipertension', 
    'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'
]

# Check if all required columns exist
available_columns = df.columns.tolist()
print(f"Available columns: {available_columns}")

# Map potential column name variations
column_mapping = {}
for col in feature_columns:
    if col in df.columns:
        column_mapping[col] = col
    else:
        # Try common variations
        variations = [col.lower(), col.upper(), col.replace('_', '')]
        for var in variations:
            if var in df.columns:
                column_mapping[col] = var
                break

print(f"Column mapping: {column_mapping}")

# Extract features and target
try:
    # Assuming the target column might be named 'No-show' or similar
    target_col = None
    possible_targets = ['No-show', 'NoShow', 'no_show', 'Show-Up']
    for target in possible_targets:
        if target in df.columns:
            target_col = target
            break
    
    if target_col is None:
        print("Warning: Target column not found automatically. Using last column as target.")
        target_col = df.columns[-1]
    
    # Extract features
    X = df[[column_mapping.get(col, col) for col in feature_columns if column_mapping.get(col, col) in df.columns]]
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns used: {X.columns.tolist()}")
    
except Exception as e:
    print(f"Error in feature extraction: {e}")
    print("Using sample data for demonstration...")
    
    # Create sample data if actual dataset is not available
    np.random.seed(42)
    n_samples = 10000
    
    X = pd.DataFrame({
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Age': np.random.randint(0, 100, n_samples),
        'Scholarship': np.random.choice([0, 1], n_samples),
        'Hipertension': np.random.choice([0, 1], n_samples),
        'Diabetes': np.random.choice([0, 1], n_samples),
        'Alcoholism': np.random.choice([0, 1], n_samples),
        'Handcap': np.random.randint(0, 4, n_samples),
        'SMS_received': np.random.choice([0, 1], n_samples)
    })
    
    y = np.random.choice(['No', 'Yes'], n_samples)
    print("Sample data created for demonstration.")

print(f"\nFeature statistics:")
print(X.describe())
print(f"\nTarget distribution:")
print(y.value_counts())

# 4. Preprocessing
print("\n4. Preprocessing...")

# Create a copy for preprocessing
X_processed = X.copy()

# Encode categorical variables
label_encoders = {}
categorical_columns = X_processed.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    label_encoders[col] = le
    print(f"Encoded column '{col}': {le.classes_}")

# Encode target variable if it's categorical
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y_processed = target_encoder.fit_transform(y)
    print(f"Target encoded: {target_encoder.classes_}")
else:
    y_processed = y.copy()

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)

print("Preprocessing completed!")
print(f"Processed features shape: {X_scaled.shape}")

# 5. Splitting the Data
print("\n5. Splitting the Data...")

# First split: 80% train, 20% temp (which will be split into 10% val, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_processed, test_size=0.2, random_state=42, stratify=y_processed
)

# Second split: Split the 20% temp into 10% validation and 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# 6. Training Decision Tree Classifiers
print("\n6. Training Decision Tree Classifiers...")

# Different criteria to test
criteria = ['gini', 'entropy']
dt_results = {}

print("Testing different criteria for Decision Tree:")
for criterion in criteria:
    dt = DecisionTreeClassifier(criterion=criterion, random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = dt.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    dt_results[criterion] = {
        'model': dt,
        'accuracy': accuracy,
        'predictions': y_val_pred
    }
    
    print(f"  {criterion.capitalize()} criterion - Validation Accuracy: {accuracy:.4f}")

# Choose best criterion
best_dt_criterion = max(dt_results.keys(), key=lambda x: dt_results[x]['accuracy'])
best_dt_model = dt_results[best_dt_criterion]['model']

print(f"\nBest Decision Tree criterion: {best_dt_criterion}")
print(f"Best Decision Tree validation accuracy: {dt_results[best_dt_criterion]['accuracy']:.4f}")

# Test the best decision tree model
y_test_pred_dt = best_dt_model.predict(X_test)
dt_test_accuracy = accuracy_score(y_test, y_test_pred_dt)
print(f"Decision Tree test accuracy: {dt_test_accuracy:.4f}")

# 7. Random Forest
print("\n7. Training Random Forest Classifiers...")

# Different numbers of estimators to test
n_estimators_list = [10, 50, 100, 200, 500]
rf_results = {}

print("Testing different numbers of estimators for Random Forest:")
for n_est in n_estimators_list:
    rf = RandomForestClassifier(
        n_estimators=n_est, 
        criterion=best_dt_criterion,  # Use best criterion from DT
        random_state=42,
        max_depth=10
    )
    rf.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    rf_results[n_est] = {
        'model': rf,
        'accuracy': accuracy,
        'predictions': y_val_pred
    }
    
    print(f"  {n_est} estimators - Validation Accuracy: {accuracy:.4f}")

# Choose best number of estimators
best_n_estimators = max(rf_results.keys(), key=lambda x: rf_results[x]['accuracy'])
best_rf_model = rf_results[best_n_estimators]['model']

print(f"\nBest Random Forest n_estimators: {best_n_estimators}")
print(f"Best Random Forest validation accuracy: {rf_results[best_n_estimators]['accuracy']:.4f}")

# Test the best random forest model
y_test_pred_rf = best_rf_model.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, y_test_pred_rf)
print(f"Random Forest test accuracy: {rf_test_accuracy:.4f}")

# 8. Final Results and Confusion Matrix
print("\n8. Final Results...")

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

print(f"\nDecision Tree (best criterion: {best_dt_criterion}):")
print(f"  Validation Accuracy: {dt_results[best_dt_criterion]['accuracy']:.4f}")
print(f"  Test Accuracy: {dt_test_accuracy:.4f}")

print(f"\nRandom Forest (best n_estimators: {best_n_estimators}):")
print(f"  Validation Accuracy: {rf_results[best_n_estimators]['accuracy']:.4f}")
print(f"  Test Accuracy: {rf_test_accuracy:.4f}")

# Determine the best overall model
if rf_test_accuracy > dt_test_accuracy:
    best_model = best_rf_model
    best_model_name = f"Random Forest (n_estimators={best_n_estimators})"
    best_predictions = y_test_pred_rf
    best_accuracy = rf_test_accuracy
else:
    best_model = best_dt_model
    best_model_name = f"Decision Tree (criterion={best_dt_criterion})"
    best_predictions = y_test_pred_dt
    best_accuracy = dt_test_accuracy

print(f"\n" + "="*50)
print("BEST MODEL RESULTS")
print("="*50)
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f}")

# Confusion Matrix for the best model
print(f"\nConfusion Matrix for {best_model_name}:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Classification Report
print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, best_predictions))

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\nFeature Importances for {best_model_name}:")
    feature_importance = pd.DataFrame({
        'feature': X_scaled.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# Summary of accuracy measures for different criteria/estimators
print("\n" + "="*50)
print("SUMMARY OF ALL TESTED PARAMETERS")
print("="*50)

print("\nDecision Tree - Different Criteria:")
for criterion, results in dt_results.items():
    print(f"  {criterion}: {results['accuracy']:.4f}")

print(f"\nRandom Forest - Different Number of Estimators:")
for n_est, results in rf_results.items():
    print(f"  {n_est} estimators: {results['accuracy']:.4f}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)