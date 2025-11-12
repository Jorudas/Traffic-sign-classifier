
import cv2
import numpy as np
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Image, TrafficSignClass


def load_training_data(limit=1000):
    db: Session = SessionLocal()

    print("📦 Krauname treniravimo duomenis iš DB...")

    images = db.query(Image).filter_by(split="train").all()
    total = len(images)

    print(f"✅ Rasta įrašų duomenų bazėje: {total}")

    X = []
    y = []

    for img in images[:limit]:
        img_data = cv2.imread(img.path)  # nuskaitome
        img_data = cv2.resize(img_data, (32, 32))  # mažinam iki 32x32
        X.append(img_data)
        y.append(img.class_id)

    X = np.array(X)
    y = np.array(y)

    from sklearn.model_selection import train_test_split

    print("✅ Vaizdai nuskaityti ir suformuoti į numpy masyvą!")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # ✂️ Padalinam į treniravimo ir testavimo duomenis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Padalinta: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test
    


if __name__ == "__main__":
    load_training_data()