
import os
import csv

from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import TrafficSignClass, Image


def ingest_training_csv_to_db():
    db: Session = SessionLocal()

    # ABSOLIUTUS KELIAS (visada veiks)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(SCRIPT_DIR, "..", "data", "GTSRB_Final_Training_Images", "GTSRB", "Final_Training", "Images")
    base_path = os.path.normpath(base_path)

    print(f"📥 Importuojame treniravimo duomenis į DB...\n🔍 Kelias: {base_path}")

    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        csv_file = os.path.join(class_path, f"GT-{class_folder}.csv")
        if not os.path.exists(csv_file):
            print(f"⚠ CSV nerastas: {csv_file}")
            continue

        class_id = int(class_folder)
        existing_class = db.query(TrafficSignClass).filter_by(id=class_id).first()
        if not existing_class:
            db.add(TrafficSignClass(id=class_id, name=None))
            db.commit()

        with open(csv_file, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader:
                filename = row[0]
                image_path = os.path.join(class_path, filename)

                db.add(Image(path=image_path, split="train", class_id=class_id))

    db.commit()
    db.close()
    print("✅ Importas baigtas — duomenys įrašyti į DB!")


if __name__ == "__main__":
    ingest_training_csv_to_db()