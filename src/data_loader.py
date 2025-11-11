
import cv2
import numpy as np
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import Image, TrafficSignClass


def load_training_data():
    db: Session = SessionLocal()

    print("📦 Krauname treniravimo duomenis iš DB...")

    images = db.query(Image).filter_by(split="train").all()
    total = len(images)

    print(f"✅ Rasta įrašų duomenų bazėje: {total}")

    X = []
    y = []

    for img in images[:1000]:  # Paimam 1000 pavyzdžių – pagreitinti testą
        img_data = cv2.imread(img.path)  # nuskaitome
        img_data = cv2.resize(img_data, (32, 32))  # mažinam iki 32x32
        X.append(img_data)
        y.append(img.class_id)

    X = np.array(X)
    y = np.array(y)

    print("✅ Vaizdai nuskaityti ir suformuoti į numpy masyvą!")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    load_training_data()