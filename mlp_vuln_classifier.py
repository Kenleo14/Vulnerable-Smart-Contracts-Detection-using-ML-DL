import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf


def configure_gpu():
    print("TensorFlow version :", tf.__version__)
    try:
        print("GPU devices        :", tf.config.list_physical_devices('GPU'))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {len(gpus)} device(s).")
            except RuntimeError as e:
                print(f"Could not set memory growth: {e}")
        else:
            print("No GPU detected, training will use CPU.")
    except Exception as e:
        print(f"Error inspecting GPUs: {e}")


def load_data(vul_path: Path, sec_path: Path) -> pd.DataFrame:
    print("\nLoading CSVs …")
    if not vul_path.exists():
        raise FileNotFoundError(f"Missing vulnerable CSV: {vul_path}")
    if not sec_path.exists():
        raise FileNotFoundError(f"Missing secure CSV: {sec_path}")

    df_vul = pd.read_csv(vul_path)
    df_sec = pd.read_csv(sec_path)

    print(f"Vulnerable rows : {len(df_vul)}")
    print(f"Secure rows     : {len(df_sec)}")

    df = pd.concat([df_vul, df_sec], ignore_index=True)
    print(f"Total rows      : {len(df)}")
    return df


def prepare_features(df: pd.DataFrame):
    # Target
    if 'label' not in df.columns:
        raise KeyError("Expected 'label' column in dataset.")

    y = df['label'].astype(int)

    # Drop everything that is not a numeric feature when present
    drop_cols = ['Unnamed: 0', 'hash_id', 'label', 'ast_nodetype', 'ast_src']
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    print(f"Feature columns : {X.shape[1]}")

    # Fill NaNs (some rows may be incomplete) → 0 and cast to float
    X = X.fillna(0).astype(float)
    return X, y


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def plot_history(history: tf.keras.callbacks.History, out_png: Path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Training curves saved → {out_png}")
    # Optional: comment out show() for headless environments
    # plt.show()
    plt.close()


def main():
    configure_gpu()

    # Paths (relative to current working directory)
    vul_csv = Path('BCCC-VolSCs-2023_Vulnerable.csv')
    sec_csv = Path('BCCC-VolSCs-2023_Secure.csv')

    df = load_data(vul_csv, sec_csv)
    X, y = prepare_features(df)

    # Train / test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train samples   : {X_train_scaled.shape[0]}")
    print(f"Test samples    : {X_test_scaled.shape[0]}")

    # Build MLP (runs on GPU automatically if available)
    model = build_model(input_dim=X_train_scaled.shape[1])

    # Callbacks – early stopping + CSV logger
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger('training_log.csv', append=False)
    ]

    # Train
    print("\n=== START TRAINING (GPU if available) ===")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,                     # early stopping will halt earlier
        batch_size=64,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on hold-out test set
    print("\n=== TEST SET EVALUATION ===")
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\nClassification Report")
    print("MLP")
    print(classification_report(y_test, y_pred,
                                target_names=['Secure (0)', 'Vulnerable (1)']))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    # Save artifacts
    out_dir = Path('.')
    model_path = out_dir / 'vulnerable_sc_mlp_gpu.h5'
    scaler_path = out_dir / 'scaler_gpu.pkl'
    curves_path = out_dir / 'training_history_gpu.png'

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel  → {model_path}")
    print(f"Scaler → {scaler_path}")

    # Plot training curves
    plot_history(history, curves_path)


if __name__ == '__main__':

    main()
