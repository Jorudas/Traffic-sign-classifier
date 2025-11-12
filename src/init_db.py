
from database import engine, Base
import models

print("🛠️ Kuriamos duomenų bazės lentelės...")
Base.metadata.create_all(bind=engine)
print("✅ Lentelės sukurtos!")

# Daliname į train/test
from database import SessionLocal
from models import Image
import random

db = SessionLocal()
images = db.query(Image).all()
random.shuffle(images)

split_point = int(len(images) * 0.8)
for i, img in enumerate(images):
    img.split = "train" if i < split_point else "test"

db.commit()
db.close()
print("✅ Duomenys suskirstyti: 80% train / 20% test")