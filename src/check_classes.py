
from database import SessionLocal
from models import Image
import numpy as np

db = SessionLocal()
images = db.query(Image).filter_by(split="train").all()
y = [img.class_id for img in images]
db.close()

print(f"Unikalių klasių: {len(set(y))}")
print("Pirmi 10 class_id:", sorted(list(set(y)))[:10])