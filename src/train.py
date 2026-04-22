import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


WINDOW_SIZE = 125
N_CHANNELS = 9
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "models/fall_detector.keras"

def load_data():
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    Y_train = np.load("data/processed/Y_train.npy")
    Y_test = np.load("data/processed/Y_test.npy")

    return X_train, X_test, Y_train, Y_test

def build_model(window_size, n_channels):
    inputs = keras.Input(shape=(window_size, n_channels))

    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)


    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    
    return model

def compute_class_weights(Y_train):
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(Y_train)
    weights = compute_class_weight('balanced', classes=classes, y=Y_train)
    class_weight_dict = dict(zip(classes, weights))

    print(f"[TRAIN] Class weights: {class_weight_dict}")
    return class_weight_dict

def train(model, X_train, Y_train, class_weights):
    os.makedirs("models", exist_ok=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Recall(name='recall'),
                 keras.metrics.Precision(name='precision')]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate(model, X_test, Y_test):
    print("\n[TRAIN] Evaluating ...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred, target_names=['No Fall', 'Fall']))

    cm = confusion_matrix(Y_test, y_pred)
    print(f"\nConfusion Matrix:")

    print(f"                 Predicted No Fall  Predicted Fall")
    print(f"  Actual No Fall       {cm[0][0]:<18} {cm[0][1]}")
    print(f"  Actual Fall          {cm[1][0]:<18} {cm[1][1]}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (caught falls)    : {tp}")
    print(f"  False Negatives (missed falls)    : {fn}  ← must be near zero")
    print(f"  False Positives (false alarms)    : {fp}")
    print(f"  True Negatives  (correct no-fall) : {tn}")

    auc = roc_auc_score(Y_test, y_pred_prob)
    print(f"\n  ROC AUC: {auc:.4f}")


def tune_threshold(model, X_test, Y_test):
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    for thresh in np.arange(0.1, 1.0, 0.1):
        y_pred = (y_pred_prob > thresh).astype(int)
        tp = ((y_pred == 1) & (Y_test == 1)).sum()
        fp = ((y_pred == 1) & (Y_test == 0)).sum()
        fn = ((y_pred == 0) & (Y_test == 1)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if(tp + fp) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) if(precision + recall) > 0 else 0)

        flag = " <- zero missed falls" if fn == 0 else ""
        print(f" {thresh:<12.1f} {recall:<10.2%} {precision:<12.2%} {f1:.4f}{flag}")

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    model = build_model(WINDOW_SIZE, N_CHANNELS)
    class_weights = compute_class_weights(Y_train)
    history = train(model, X_train, Y_train, class_weights)

    model = keras.models.load_model(MODEL_PATH)
    evaluate(model, X_test, Y_test)
    tune_threshold(model, X_test, Y_test)

    print(f"\n[TRAIN] Model saved to {MODEL_PATH}")
    print(f"[TRAIN] Ready for import")

