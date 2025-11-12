
# src/test_db_view.py
from database import SessionLocal
from models import Image, TrafficSignClass

db = SessionLocal()

print("🧭 Lentelė 'classes':")
for c in db.query(TrafficSignClass).limit(5):
    print(f"ID={c.id}, Name={c.name}")

print("\n🖼️ Lentelė 'images':")
for i in db.query(Image).limit(5):
    print(f"Path={i.path}, Class={i.class_ref.name}, Split={i.split}")

db.close()