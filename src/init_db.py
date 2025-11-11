
from .database import engine, Base
from . import models

print("🔧 Kuriamos duomenų bazės lentelės...")
Base.metadata.create_all(bind=engine)
print("✅ Lentelės sukurtos!")