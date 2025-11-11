
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from .database import SessionLocal
from .models import Image


def load_data(limit=5000):
    db: Session = SessionLocal()

    print("📦 Krauname duomenis modeliui...")

    images = db.query(Image).filter_by(split="train").all()
    total = len(images)
    print(f"✅ Rasta treniravimo įrašų: {total}")

    X = []
    y = []

    for img in images[:limit]:
        import cv2
        data = cv2.imread(img.path)
        data = cv2.resize(data, (32, 32))
        X.append(data)
        y.append(img.class_id)

    X = np.array(X) / 255.0
    y = to_categorical(np.array(y), num_classes=43)

    print(f"✅ Suformuota X: {X.shape}, y: {y.shape}")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model():
    X_train, X_test, y_train, y_test = load_data(limit=5000)

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

    print("🚀 Pradedame mokymą...")
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    print("✅ Mokymas baigtas!")
    model.save("traffic_sign_cnn.h5")
    print("💾 Modelis išsaugotas: traffic_sign_cnn.h5")


if __name__ == "__main__":
    train_model()