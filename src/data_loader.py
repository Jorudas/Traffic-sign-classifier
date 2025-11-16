# src/data_loader.py
import os
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split

from database import SessionLocal
from models import GTSRBRecord


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "GTSRB_Final_Training_Images",
    "GTSRB",
    "Final_Training",
    "Images",
)

TEST_ROOT = os.path.join(
    BASE_DIR,
    "data",
    "GTSRB_Final_Test_Images",
    "GTSRB",
    "Final_Test",
    "Images",
)


def _load_split_from_db(split: str, limit: int | None, image_size=(32, 32),
                        flatten: bool = False):
    db: Session = SessionLocal()
    query = db.query(GTSRBRecord).filter(GTSRBRecord.split == split)

    if limit is not None:
        query = query.limit(limit)

    rows = query.all()
    db.close()

    X = []
    y = []

    for row in rows:
        if split == "train":
            img_dir = os.path.join(TRAIN_ROOT, f"{row.class_id:05d}")
        else:
            img_dir = TEST_ROOT

        img_path = os.path.join(img_dir, row.filename)

        if not os.path.exists(img_path):
            # jeigu failo nėra – praleidžiam
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Iškerpam ROI
        img = img.crop((row.roi_x1, row.roi_y1, row.roi_x2, row.roi_y2))
        img = img.resize(image_size)

        arr = np.array(img, dtype=np.float32) / 255.0

        if flatten:
            arr = arr.reshape(-1)

        X.append(arr)
        y.append(row.class_id)

    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    return X, y


def load_data_for_ml(limit: int | None = 5000):
    """
    ML modeliui (RandomForest): grąžina X_flat (N, D), y.

    - ima tik train split
    - flatten = True
    """
    X, y = _load_split_from_db("train", limit=limit, image_size=(32, 32), flatten=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_data_for_cnn(limit: int | None = 8000):
    """
    CNN modeliui:
    - grąžina X_train, X_val, y_train, y_val
    """
    X, y = _load_split_from_db("train", limit=limit, image_size=(32, 32), flatten=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # į one-hot
    num_classes = len(np.unique(y))
    y_train_oh = np.eye(num_classes)[y_train]
    y_val_oh = np.eye(num_classes)[y_val]

    return X_train, X_val, y_train_oh, y_val_oh, num_classes