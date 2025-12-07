import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load Dataset
def load_data(filepath):
    """Load the dataset and remove the leaky 'Unnamed: 0' index column."""
    print(f"Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        
        # --- THIS IS THE FIX: Drop the leaky index column ---
        if 'Unnamed: 0' in df.columns:
            print("Found and dropping leaky 'Unnamed: 0' column.")
            df = df.drop(columns=['Unnamed: 0'])
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    return df
    
# Preprocess Data for Classical Models
def preprocess_data(df, target_column):
    """Select only numeric features, separate from target, and split into train/test sets."""
    y = df[target_column]
    features_df = df.drop(columns=[target_column])
    X = features_df.select_dtypes(include=np.number)
    
    print(f"Original feature columns: {len(features_df.columns)}")
    print(f"Selected numeric feature columns for training: {len(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Testing set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Train Random Forest Model
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train a Random Forest classifier."""
    print("\nTraining Random Forest Classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    print("Training complete!")
    
    return rf_model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (POST-FIX)")
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

# Feature Importance
def get_feature_importance(model, feature_names, top_n=10):
    """Get and display top N important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop {top_n} Most Important Features (Post-Fix):")
    print("-" * 40)
    for i in range(min(top_n, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
# Save Model
def save_model(model, scaler, model_path='random_forest_model.pkl', scaler_path='rf_scaler.pkl'):
    """Save the trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

# Main Execution
if __name__ == "__main__":
    DATASET_PATH = "full_contracts_dataset_with_source.csv"
    TARGET_COLUMN = "label"
    
    df = load_data(DATASET_PATH)
    if df is not None:
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, TARGET_COLUMN)
        
        # Train model
        rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=20)
        
        # Evaluate model
        evaluate_model(rf_model, X_test, y_test)
        
        # Get feature importance
        get_feature_importance(rf_model, feature_names, top_n=10)
        
        # Save model
        save_model(rf_model, scaler)

