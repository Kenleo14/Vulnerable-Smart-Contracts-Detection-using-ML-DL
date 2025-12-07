
#   XGBoost

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import tensorflow as tf

print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))

print("\nLoading datasets...")
df_vul = pd.read_csv('BCCC-VolSCs-2023_Vulnerable.csv')
df_sec = pd.read_csv('BCCC-VolSCs-2023_Secure.csv')
print(f"Vulnerable : {len(df_vul)}")
print(f"Secure     : {len(df_sec)}")
df = pd.concat([df_vul, df_sec], ignore_index=True)
print(f"Total      : {len(df)}")

y = df['label'].astype(int)
drop_cols = ['Unnamed: 0', 'hash_id', 'label', 'ast_nodetype', 'ast_src']
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols).fillna(0).astype(float)
print(f"Features   : {X.shape[1]}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, auc_scores, accuracies = [], [], []
all_y_true, all_y_pred, all_y_pred_proba = [], [], []

# Class imbalance ratio
neg, pos = y.value_counts()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

print("\n" + "="*60)
print("5-FOLD CV (Optimized XGBoost)")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/5")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
    dval = xgb.DMatrix(X_val_sc, label=y_val)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 10,           # Increased
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'tree_method': 'hist'
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    y_pred_proba = bst.predict(dval)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    f1 = f1_score(y_val, y_pred, average='binary')
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    f1_scores.append(f1)
    auc_scores.append(auc)
    accuracies.append(acc)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_pred_proba.extend(y_pred_proba)

    print(f"  F1 : {f1:.4f} | AUC : {auc:.4f} | Acc : {acc:.4f} | Best : {bst.best_iteration}")

print("\n" + "="*60)
print("OVERALL RESULTS")
print("="*60)
print(f"F1 (vuln) : {f1_score(all_y_true, all_y_pred, average='binary'):.4f}")
print(f"AUC       : {roc_auc_score(all_y_true, all_y_pred_proba):.4f}")
print(f"Accuracy  : {accuracy_score(all_y_true, all_y_pred):.4f}")
print(classification_report(all_y_true, all_y_pred, target_names=['Secure', 'Vulnerable']))

print("\nTraining final model...")
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
dtrain_full = xgb.DMatrix(X_res, label=y_res)

final_bst = xgb.train(params, dtrain_full, num_boost_round=1000)
final_bst.save_model('vulnerable_sc_xgb_final.model')

# SHAP: Reload + set base_score
final_bst = xgb.Booster()
final_bst.load_model('vulnerable_sc_xgb_final.model')
final_bst.set_attr(base_score='0.5')  # Critical fix
final_bst.save_model('vulnerable_sc_xgb_final_fixed.model')

joblib.dump(final_scaler, 'scaler_final.pkl')
print("Model → vulnerable_sc_xgb_final_fixed.model")
print("Scaler → scaler_final.pkl")

print("\nSHAP plot...")
explainer = shap.TreeExplainer(final_bst)
shap_vals = explainer.shap_values(X_scaled[:500])
shap.summary_plot(shap_vals, X.iloc[:500], max_display=20)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
plt.show()

print("\nPER-FOLD F1:")
for i, f in enumerate(f1_scores, 1):
    print(f"Fold {i}: {f:.4f}")
print(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
