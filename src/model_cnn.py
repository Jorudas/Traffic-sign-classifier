
import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 🔇 Slopiname perteklinius pranešimus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# 🧩 Duomenų užkrovimas
from database import SessionLocal
from data_loader import load_training_data


def train_model():
    # 1️⃣ Įkeliame duomenis iš DB
    print("📦 Krauname treniravimo duomenis iš DB...")
    X_train, X_test, y_train, y_test = load_training_data(limit=5000)

    # 2️⃣ Konvertuojame žymes į one-hot formatą
    y_train = to_categorical(y_train, num_classes=43)
    y_test = to_categorical(y_test, num_classes=43)

    # 3️⃣ Kuriame CNN modelį
    print("🧠 Kuriame CNN modelį...")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(43, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 4️⃣ Mokymas
    print("🚀 Pradedame mokymą...")
    history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

    # 🧠 ĮVERTINIMAS PO MOKYMO
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\n🎯 Galutinis modelio tikslumas testavimo duomenyse: {test_acc * 100:.2f}%")
    print(f"📉 Galutinis testavimo nuostolis (loss): {test_loss:.4f}")

    # 5️⃣ Išsaugome modelį
    print("✅ Mokymas baigtas!")
    model.save("traffic_sign_cnn.h5")
    model.save("traffic_sign_cnn.keras")
    print("💾 Modelis sėkmingai išsaugotas!")

    # 6️⃣ Nubraižome tikslumo grafiką
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Tikslumas (train)')
    plt.plot(history.history['val_accuracy'], label='Tikslumas (val)')
    plt.title('CNN modelio tikslumas')
    plt.xlabel('Epoka')
    plt.ylabel('Tikslumas')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_model()