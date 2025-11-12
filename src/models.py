# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


# Lentelė kelių ženklų klasėms
class TrafficSignClass(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)

    # Ryšys su Image lentele (viena klasė -> daug nuotraukų)
    images = relationship("Image", back_populates="class_ref")


# Lentelė su nuotraukų informacija
class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)

    # Kur yra failas (kelias iki nuotraukos)
    path = Column(String, nullable=False)

    # Train / Test / UserTrain / UserTest
    split = Column(String, nullable=False)

    # Priskirta klasė (gali būti NULL, jei naudotojo įkelta test nuotrauka)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=True)

    # Kada įrašyta
    created_at = Column(DateTime, default=datetime.utcnow)

    # Ryšys: ši nuotrauka turi klasę
    class_ref = relationship("TrafficSignClass", back_populates="images")