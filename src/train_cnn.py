
# src/train_cnn.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from data_loader import load_data_for_cnn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "traffic_sign_cnn_simple.h5")


# ============================================================
# 1️⃣  CNN modelio architektūra
# ============================================================
def build_cnn(input_shape, num_classes: int) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 2️⃣  Modelio treniravimas
# ============================================================
def main():
    print("📦 Krauname duomenis CNN modeliui...")
    X_train, X_val, y_train, y_val, num_classes = load_data_for_cnn(limit=8000)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, classes: {num_classes}")

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

    ckpt = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[ckpt],
    )

    print("✅ Treniravimas baigtas. Modelis išsaugotas:", MODEL_PATH)

    # ============================================================
    # 3️⃣  Istorijos išsaugojimas į .npz (Streamlit naudosis)
    # ============================================================
    np.savez(
        os.path.join(BASE_DIR, "cnn_history_simple.npz"),
        loss=np.array(history.history["loss"]),
        val_loss=np.array(history.history["val_loss"]),
        acc=np.array(history.history["accuracy"]),
        val_acc=np.array(history.history["val_accuracy"]),
    )
    print("📁 Išsaugota: cnn_history_simple.npz")

    # ============================================================
    # 4️⃣  Grafikų išsaugojimas į PNG egzaminui
    # ============================================================

    # --- Accuracy grafikas ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Val accuracy")
    plt.title("CNN Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, "src", "cnn_accuracy.png"))
    plt.close()

    # --- Loss grafikas ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("CNN Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, "src", "cnn_loss.png"))
    plt.close()

    print("📊 Grafikai išsaugoti: cnn_accuracy.png, cnn_loss.png")


if __name__ == "__main__":
    main()