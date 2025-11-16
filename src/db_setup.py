from database import Base, engine
from models import GTSRBRecord    # ← ŠITAS PRIVALO BŪTI

if __name__ == "__main__":
    print("🛠️ Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Done!")