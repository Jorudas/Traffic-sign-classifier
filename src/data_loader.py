
# src/data_loader.py
import os
import cv2
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from database import SessionLocal
from models import Image
from preprocess import preprocess_image


def load_training_data(limit=None, target_size=(64, 64)):
    """
    Nuskaito treniravimo duomenis iš duomenų bazės (GTSRB dataset),
    apdoroja per preprocess_image() ir padalina į train/test rinkinius.
    """

    db: Session = SessionLocal()
    print("📦 Krauname treniravimo duomenis iš DB...")

    images = db.query(Image).filter_by(split="train").all()
    total = len(images)
    print(f"✅ Rasta įrašų duomenų bazėje: {total}")

    X, y = [], []

    for i, img in enumerate(images if limit is None else images[:limit]):
        if not os.path.exists(img.path):
            print(f"⚠️ Nerastas failas: {img.path}")
            continue

        try:
            img_data = preprocess_image(img.path, target_size=target_size)
            X.append(img_data)
            y.append(img.class_id)
        except Exception as e:
            print(f"❌ Klaida nuskaitant {img.path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    db.close()

    print(f"✅ Vaizdai apdoroti: {X.shape}, Žymės: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Padalinta: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_data_generators(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Grąžina duomenų generatorius su augmentacija.
    Augmentacija padidina modelio atsparumą realioms Google nuotraukoms.
    """

    train_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.6, 1.4],
        rescale=1./255,
        fill_mode='nearest'
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_flow = val_gen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

    print("✅ Sukurti duomenų generatoriai su augmentacija.")
    return train_flow, val_flow


if __name__ == "__main__":
    # Greitas testas – pamatyti ar viskas veikia
    X_train, X_test, y_train, y_test = load_training_data(limit=100)
    print("✅ Testinis duomenų įkėlimas pavyko.")