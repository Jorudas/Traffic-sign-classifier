
# src/ingest_gtsrb_to_db.py
import os
import pandas as pd
from database import SessionLocal, Base, engine
from models import TrafficSignClass, Image

# Keliai iki duomenų
CSV_PATH = "../data/GTSRB_Final_Test_GT/GT-final_test.csv"
IMAGES_DIR = "../data/GTSRB_Final_Test_Images/"

def ingest_data(limit=20):
    # Sukuriame lenteles, jei jų dar nėra
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    print("📂 Nuskaitome CSV...")
    df = pd.read_csv(CSV_PATH, sep=';')

    # Sukuriame / atnaujiname klases
    existing_classes = {c.name: c for c in db.query(TrafficSignClass).all()}
    unique_classes = sorted(df["ClassId"].unique())

    for cid in unique_classes:
        cname = f"class_{cid}"
        if cname not in existing_classes:
            cls = TrafficSignClass(name=cname)
            db.add(cls)
            db.commit()
            existing_classes[cname] = cls

    print(f"✅ Pridėtos / atnaujintos {len(unique_classes)} klasės.")

    # Įrašome keletą pavyzdžių iš CSV
    print("🖼️ Įrašome kelis pavyzdinius vaizdus į DB...")
    count = 0
    for _, row in df.iterrows():
        if count >= limit:
            break
        img_path = os.path.join(IMAGES_DIR, row["Filename"])
        class_id = row["ClassId"]
        class_ref = db.query(TrafficSignClass).filter_by(name=f"class_{class_id}").first()

        image = Image(
            path=img_path,
            split="test",  # testiniai vaizdai
            class_ref=class_ref
        )
        db.add(image)
        count += 1

    db.commit()
    db.close()
    print(f"✅ Į DB įrašyta {count} nuotraukų.")

if __name__ == "__main__":
    ingest_data()