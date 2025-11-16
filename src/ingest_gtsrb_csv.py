
# src/ingest_gtsrb_csv.py
import os
import csv
from sqlalchemy.orm import Session

from database import SessionLocal
from models import GTSRBRecord


# ============================================================
# ⛽ CSV → DB importavimo funkcija (TRAIN + TEST anotacijos)
# ============================================================

def ingest_gtsrb_train(train_dir: str):
    """
    Sukelia TRAIN anotacijas į DB.

    Tikimasi standartinės GTSRB struktūros:
    data/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/
        00000/
            GT-00000.csv
            00000_00000.ppm
            ...

    Kiekviename klasės aplanke yra GT-<klase>.csv failas.
    """
    db: Session = SessionLocal()

    total = 0
    for class_folder in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, f"GT-{class_folder}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Nerastas CSV: {csv_path}")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                rec = GTSRBRecord(
                    filename=row["Filename"],
                    width=int(row["Width"]),
                    height=int(row["Height"]),
                    roi_x1=int(row["Roi.X1"]),
                    roi_y1=int(row["Roi.Y1"]),
                    roi_x2=int(row["Roi.X2"]),
                    roi_y2=int(row["Roi.Y2"]),
                    class_id=int(row["ClassId"]),
                    split="train",
                )
                db.add(rec)
                total += 1

    db.commit()
    db.close()
    print(f"✅ Train anotacijos suimportuotos (įrašų: {total})")


def ingest_gtsrb_test(test_csv_path: str):
    """
    Sukelia TEST anotacijas iš GT-final_test.csv į DB.
    """
    db: Session = SessionLocal()

    total = 0
    with open(test_csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            rec = GTSRBRecord(
                filename=row["Filename"],
                width=int(row["Width"]),
                height=int(row["Height"]),
                roi_x1=int(row["Roi.X1"]),
                roi_y1=int(row["Roi.Y1"]),
                roi_x2=int(row["Roi.X2"]),
                roi_y2=int(row["Roi.Y2"]),
                class_id=int(row["ClassId"]),
                split="test",
            )
            db.add(rec)
            total += 1

    db.commit()
    db.close()
    print(f"✅ Test anotacijos suimportuotos (įrašų: {total})")


# ============================================================
# 🔥 Paleidimas per terminalą
# ============================================================

if __name__ == "__main__":
    train_path = os.path.join(
        "data", "GTSRB_Final_Training_Images", "GTSRB", "Final_Training", "Images"
    )
    test_csv = os.path.join("data", "GTSRB_Final_Test_GT", "GT-final_test.csv")

    ingest_gtsrb_train(train_path)
    ingest_gtsrb_test(test_csv)

    print("🎉 VISOS CSV ANOTACIJOS SUIMPORTUOTOS Į DB")