
# src/check_db_preview.py
import random
import cv2
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Image, TrafficSignClass


def show_random_images(n=20):
    db: Session = SessionLocal()

    print("📂 Nuskaitome visą DB...")
    images = db.query(Image).filter_by(split="train").all()

    print(f"🔢 Iš viso DB įrašų: {len(images)}")

    print("\n🧪 Rodysiu atsitiktinius paveikslėlius...")

    for i in range(n):
        img = random.choice(images)

        print(f"\n---------------")
        print(f"🔍 KLASĖ (class_id): {img.class_id}")

        # nuskaitom paveikslėlį
        image = cv2.imread(img.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # rodom per matplotlib
        plt.imshow(image)
        plt.title(f"class_id = {img.class_id}")
        plt.axis("off")
        plt.show()

        input("👉 Paspausk ENTER, kad rodyti kitą...")

    print("\n✅ Patikrinimas baigtas!")


if __name__ == "__main__":
    show_random_images(20)