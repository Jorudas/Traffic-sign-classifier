
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Sukuriame bazinį SQLAlchemy klasės pagrindą
Base = declarative_base()

# Duomenų bazės kelias (SQLite)
DATABASE_URL = "sqlite:///db/gtsrb.db"

# Engine už duomenų bazės ryšį
engine = create_engine(DATABASE_URL, echo=True)

# Sesija, per kurią galėsime atlikti INSERT, SELECT, UPDATE
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)