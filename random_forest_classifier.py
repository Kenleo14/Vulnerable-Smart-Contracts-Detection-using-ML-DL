"""
Random Forest Classifier for Vulnerable Smart Contracts Detection
Dataset: BCCC-VulSCs-2023 (36,670 samples, 70 features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================
# 1. Load Dataset
# ============================================
def load_data(filepath):
    """Load the dataset from a CSV file."""
    print(f"Loading dataset from: {filepath}")
    # Handle potential parsing errors in large CSVs
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Add fallback for common issues if needed, e.g., different encoding
        # df = pd.read_csv(filepath, encoding='latin1')
        return None
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    return df

# ============================================
# 2. Preprocess Data for Classical Models
# ============================================
def preprocess_data(df, target_column, text_column_to_drop):
    """Separate features and target, then split into train/test sets."""
    # Drop the target column AND the raw source code text column
    print(f"Dropping '{target_column}' and '{text_column_to_drop}' columns to create feature set.")
    X = df.drop(columns=[target_column, text_column_to_drop], errors='ignore')
    y = df[target_column]
    
    # Verify we have the expected number of feature columns (around 70)
    print(f"Number of feature columns: {X.shape[1]}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Testing set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# ============================================
# 3. Train Random Forest Model
# ============================================
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest classifier."""
    print("\nTraining Random Forest Classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    rf_model.fit(X_train, y_train)
    print("Training complete!")
    
    return rf_model

# ============================================
# 4. Evaluate Model
# ============================================
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred

# ============================================
# 5. Feature Importance
# ============================================
def get_feature_importance(model, feature_names, top_n=10):
    """Get and display top N important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 40)
    for i in range(min(top_n, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================
# 6. Save Model
# ============================================
def save_model(model, scaler, model_path='random_forest_model.pkl', scaler_path='rf_scaler.pkl'):
    """Save the trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    # --- Configuration based on bert_vuln_smart_contracts.py ---
    DATASET_PATH = "full_contracts_dataset_with_source.csv"
    TARGET_COLUMN = "label"
    # This is the text column used by BERT, which we need to drop for classical models
    TEXT_COLUMN_TO_DROP = "source_code"
    
    # Load and preprocess data
    df = load_data(DATASET_PATH)
    if df is not None:
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, TARGET_COLUMN, TEXT_COLUMN_TO_DROP)
        
        # Train model
        rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=20)
        
        # Evaluate model
        evaluate_model(rf_model, X_test, y_test)
        
        # Get feature importance
        get_feature_importance(rf_model, feature_names, top_n=10)
        
        # Save model
        save_model(rf_model, scaler)
