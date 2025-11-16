
# src/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base


class GTSRBRecord(Base):
    """
    Anotacijų lentelė (train + test).
    Užpildoma per ingest_gtsrb_csv.py
    """
    __tablename__ = "gtsrb_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)   # pvz. "00000_00000.ppm"
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    roi_x1 = Column(Integer, nullable=False)
    roi_y1 = Column(Integer, nullable=False)
    roi_x2 = Column(Integer, nullable=False)
    roi_y2 = Column(Integer, nullable=False)
    class_id = Column(Integer, nullable=False)  # 0–42
    split = Column(String, nullable=False)      # "train" arba "test"


class ModelResult(Base):
    """
    Modelio spėjimų rezultatai (egzamino reikalavimas: saugoti į DB).
    Pildoma iš Streamlit aplikacijos arba treniravimų metu.
    """
    __tablename__ = "prediction_results"  # paliekam tą pačią lentelę DB

    id = Column(Integer, primary_key=True, index=True)

    # iš kur atkeliavo nuotrauka: failo kelias arba "upload" arba URL
    image_source = Column(String, nullable=False)

    # kokį modelį naudojome: "cnn", "mobilenet", "rf", "knn" ir t.t.
    model_type = Column(String, nullable=False)

    # jei žinome teisingą label (pvz. test rinkinyje)
    true_class = Column(Integer, nullable=True)

    # modelio pateiktas spėjimas
    predicted_class = Column(Integer, nullable=False)

    # modelio tikimybė (MobileNet ir CNN turi)
    probability = Column(Float, nullable=True)

    # ar čia train/test/user įrašas (papildoma informacija)
    split = Column(String, nullable=True)  # "train", "test", "user"

    # rezultatų saugojimo laikas
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelResult(model={self.model_type}, pred={self.predicted_class})>"