import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample medical appointment data for demonstration"""
    print("Dataset file not found. Creating sample data for demonstration...")
    print("To use real data, download from:")
    print("https://www.kaggle.com/joniarroba/noshowappointments")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Create sample data
    data = {
        'Gender': np.random.choice(['M', 'F'], n_samples, p=[0.35, 0.65]),
        'Age': np.random.randint(0, 95, n_samples),
        'Neighbourhood': np.random.choice([f'Area_{i}' for i in range(1, 21)], n_samples),
        'Scholarship': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Hypertension': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Diabetes': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Alcoholism': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        'Handicap': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.95, 0.03, 0.015, 0.004, 0.001]),
        'SMS_received': np.random.choice([0, 1], n_samples, p=[0.32, 0.68]),
    }
    
    # Create target variable with some logic
    no_show_prob = 0.2  # Base probability
    for i in range(n_samples):
        # Adjust probability based on features
        prob = no_show_prob
        if data['Age'][i] < 18:
            prob += 0.05
        if data['Age'][i] > 65:
            prob -= 0.03
        if data['SMS_received'][i] == 0:
            prob += 0.08
        if data['Scholarship'][i] == 1:
            prob += 0.05
        
        prob = max(0, min(1, prob))  # Keep probability between 0 and 1
        data.setdefault('No-show', []).append(np.random.choice([0, 1], p=[1-prob, prob]))
    
    df = pd.DataFrame(data)
    print(f"Sample dataset created! Shape: {df.shape}")
    return df

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    try:
        # Try to load the actual dataset
        df = pd.read_csv('KaggleV2-May-2016.csv')
        print("Loaded real dataset from file!")
    except FileNotFoundError:
        # Create sample data if file not found
        df = create_sample_data()
    
    print("\n=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Check for missing values
    print(f"\n=== MISSING VALUES ===")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found!")
    
    return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n=== DATA PREPROCESSING ===")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle different possible target column names
    target_cols = ['No-show', 'NoShow', 'no_show', 'target']
    target_col = None
    for col in target_cols:
        if col in df_processed.columns:
            target_col = col
            break
    
    if target_col is None:
        print("Warning: No target column found. Using last column as target.")
        target_col = df_processed.columns[-1]
    
    print(f"Using '{target_col}' as target variable")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != target_col:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Handle target variable if it's categorical
    if df_processed[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df_processed[target_col] = le_target.fit_transform(df_processed[target_col])
        label_encoders['target'] = le_target
    
    # Feature engineering
    if 'Age' in df_processed.columns:
        df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                         bins=[0, 18, 35, 50, 65, 100], 
                                         labels=['Child', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
        df_processed['Age_Group'] = LabelEncoder().fit_transform(df_processed['Age_Group'])
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
    
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Target variable distribution:\n{df_processed[target_col].value_counts()}")
    
    return df_processed, target_col, label_encoders

def perform_eda(df, target_col):
    """Perform Exploratory Data Analysis"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    plt.figure(figsize=(15, 10))
    
    # Target distribution
    plt.subplot(2, 3, 1)
    df[target_col].value_counts().plot(kind='bar')
    plt.title('Target Variable Distribution')
    plt.xlabel('No-Show (0=Show, 1=No-Show)')
    plt.ylabel('Count')
    
    # Age distribution
    if 'Age' in df.columns:
        plt.subplot(2, 3, 2)
        plt.hist(df['Age'], bins=30, edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
    
    # Correlation heatmap
    plt.subplot(2, 3, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    
    # Gender vs No-show (if Gender exists)
    if 'Gender' in df.columns:
        plt.subplot(2, 3, 4)
        gender_noshow = pd.crosstab(df['Gender'], df[target_col])
        gender_noshow.plot(kind='bar')
        plt.title('Gender vs No-Show')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(['Show', 'No-Show'])
    
    # SMS vs No-show (if SMS_received exists)
    if 'SMS_received' in df.columns:
        plt.subplot(2, 3, 5)
        sms_noshow = pd.crosstab(df['SMS_received'], df[target_col])
        sms_noshow.plot(kind='bar')
        plt.title('SMS Received vs No-Show')
        plt.xlabel('SMS Received (0=No, 1=Yes)')
        plt.ylabel('Count')
        plt.legend(['Show', 'No-Show'])
    
    # Feature importance preview
    plt.subplot(2, 3, 6)
    feature_cols = [col for col in df.columns if col != target_col]
    X_temp = df[feature_cols]
    y_temp = df[target_col]
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_temp, y_temp)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nTarget variable '{target_col}' distribution:")
    print(df[target_col].value_counts(normalize=True))
    
    return feature_importance

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print("\n=== MODEL TRAINING ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name} - ROC AUC: {roc_auc}")
        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_models(results, X_test, y_test):
    """Evaluate and compare model performance"""
    print("\n=== MODEL EVALUATION ===")
    
    plt.figure(figsize=(15, 10))
    
    # ROC Curves
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    
    # Model Performance Comparison
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    roc_scores = [results[name]['roc_auc'] if isinstance(results[name]['roc_auc'], float) else 0 
                  for name in model_names]
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, roc_scores, width, label='ROC AUC', alpha=0.8)
    plt.bar(x + width/2, cv_scores, width, label='CV Accuracy', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    
    # Best model confusion matrix
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['roc_auc'] if isinstance(results[x]['roc_auc'], float) else 0)
    
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Feature importance for best model
    plt.subplot(2, 2, 4)
    best_model = results[best_model_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        feature_names = [f'Feature_{i}' for i in range(len(best_model.feature_importances_))]
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        plt.bar(range(len(indices)), importances[indices])
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n=== DETAILED RESULTS ===")
    for name, result in results.items():
        print(f"\n{name}:")
        print(classification_report(y_test, result['predictions']))
    
    print(f"\nBest performing model: {best_model_name}")
    return best_model_name, results[best_model_name]

def hyperparameter_tuning(X_train, y_train, model_name):
    """Perform hyperparameter tuning for the best model"""
    print(f"\n=== HYPERPARAMETER TUNING FOR {model_name.upper()} ===")
    
    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    else:
        print(f"Hyperparameter tuning not implemented for {model_name}")
        return None
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    """Main function to run the complete analysis"""
    print("=== MEDICAL APPOINTMENT NO-SHOW PREDICTION ===")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Preprocess data
    df_processed, target_col, label_encoders = preprocess_data(df)
    
    # Perform EDA
    feature_importance = perform_eda(df_processed, target_col)
    
    # Prepare features and target
    feature_cols = [col for col in df_processed.columns if col != target_col]
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate models
    best_model_name, best_result = evaluate_models(results, X_test_scaled, y_test)
    
    # Hyperparameter tuning for best model
    tuned_model = hyperparameter_tuning(X_train_scaled, y_train, best_model_name)
    
    if tuned_model:
        print("\n=== FINAL TUNED MODEL PERFORMANCE ===")
        y_pred_tuned = tuned_model.predict(X_test_scaled)
        y_pred_proba_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"Tuned {best_model_name} ROC AUC: {roc_auc_score(y_test, y_pred_proba_tuned):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_tuned))
    
    print("\n=== ANALYSIS COMPLETE ===")
    return {
        'best_model': best_model_name,
        'results': results,
        'tuned_model': tuned_model if tuned_model else best_result['model'],
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_col': target_col
    }

if __name__ == "__main__":
    results = main()