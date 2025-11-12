from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

Base = declarative_base()

# 🔧 Dinamiškas kelias (automatiškai veiks tiek vietoje, tiek Colab)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "db", "gtsrb.db")

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# 🧩 Sukuriame engine
engine = create_engine(DATABASE_URL, echo=True)

# 🧠 Sesija (SQLAlchemy)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)