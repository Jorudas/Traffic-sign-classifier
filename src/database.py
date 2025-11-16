# src/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Bazinė direktorija (CLEAN_REPO/src → CLEAN_REPO)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DB failas: data/db/gtsrb.db
DB_PATH = os.path.join(BASE_DIR, "data", "db", "gtsrb.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()