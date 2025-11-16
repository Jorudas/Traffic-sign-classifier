
# src/app.py
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st

from tensorflow.keras.models import load_model
from sqlalchemy.orm import Session
from sqlalchemy import func

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from database import SessionLocal
from models import GTSRBRecord, ModelResult
from labels import CLASS_LABELS


# =========================================
#  🔧 Pagalbinės funkcijos
# =========================================

def preprocess_for_model(img: Image.Image, model_name: str) -> np.ndarray:
    """Paruošiam paveikslą CNN arba MobileNet modeliui."""
    if model_name == "MobileNet":
        target_size = (224, 224)
    else:
        target_size = (32, 32)

    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@st.cache_resource
def load_model_cached(model_name: str):
    """Įkeliame pasirinktą modelį (su cache)."""
    try:
        if model_name == "MobileNet":
            model_path = os.path.join("src", "mobilenet_gtsrb_best.h5")
        else:
            model_path = os.path.join("src", "traffic_sign_cnn_best.h5")

        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Klaida įkeliant {model_name} modelį: {e}")
        return None


def get_random_test_record() -> GTSRBRecord | None:
    """Atsitiktinė TEST įrašo eilutė – spėjimui be įkėlimo."""
    db: Session = SessionLocal()
    try:
        rec = (
            db.query(GTSRBRecord)
            .filter(GTSRBRecord.split == "test")
            .order_by(func.random())
            .first()
        )
        return rec
    finally:
        db.close()


def load_image_from_record(rec: GTSRBRecord) -> Image.Image:
    """
    Nuskaitom nuotrauką iš disko pagal DB įrašą ir iškerpam ROI.
    Bandome kelis galimus kelių variantus, kad būtų saugu.
    """
    candidates = [
        os.path.join("data", "GTSRB_Final_Test_Images", rec.filename),
        os.path.join(
            "data",
            "GTSRB_Final_Training_Images",
            "GTSRB",
            "Final_Training",
            "Images",
            rec.filename,
        ),
        os.path.join("data", rec.filename),
    ]

    img = None
    for path in candidates:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            break

    if img is None:
        raise FileNotFoundError(f"Failas nerastas (filename={rec.filename})")

    roi = (rec.roi_x1, rec.roi_y1, rec.roi_x2, rec.roi_y2)
    img = img.crop(roi)
    return img


def save_user_image_to_db(img: Image.Image, class_id: int, split: str) -> None:
    """
    Vartotojo įkelta nuotrauka:
    - išsaugoma į data/user_images
    - sukuriamas GTSRBRecord DB įrašas (split='user_train' ar 'user_test')
    """
    os.makedirs(os.path.join("data", "user_images"), exist_ok=True)

    filename = f"user_{split}_{random.randint(100000, 999999)}.png"
    rel_path = os.path.join("user_images", filename)
    full_path = os.path.join("data", rel_path)
    img.save(full_path)

    db: Session = SessionLocal()
    try:
        rec = GTSRBRecord(
            filename=rel_path,
            width=img.width,
            height=img.height,
            roi_x1=0,
            roi_y1=0,
            roi_x2=img.width,
            roi_y2=img.height,
            class_id=class_id,
            split=split,
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()


def save_prediction_result(
    image_source: str,
    model_name: str,
    true_class: int | None,
    predicted_class: int,
    split: str | None,
) -> None:
    """Įrašom modelio spėjimą į ModelResult lentelę."""
    db: Session = SessionLocal()
    try:
        row = ModelResult(
            image_source=image_source,
            model_name=model_name,
            true_class=true_class,
            predicted_class=predicted_class,
            split=split,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


# =========================================
#  🖥️ Streamlit UI
# =========================================

st.set_page_config(
    page_title="GTSRB kelio ženklų klasifikatorius (CNN + MobileNet)",
    layout="wide",
)

st.title("🚦 GTSRB kelio ženklų klasifikatorius (CNN + MobileNet)")

tab_upload, tab_graphs, tab_db = st.tabs(
    ["📤 Įkelti ženklą / mokymas", "📈 Grafikai ir metrikos", "📂 DB pavyzdžiai"]
)

# =========================================
# 1️⃣ Įkėlimas + spėjimas + mokymas
# =========================================

with tab_upload:
    st.subheader("1. Pasirink modelį")

    model_name = st.radio(
        "Modelio tipas:",
        ["CNN (32×32)", "MobileNet"],
        horizontal=True,
    )

    internal_model_name = "MobileNet" if model_name == "MobileNet" else "CNN"
    model = load_model_cached(internal_model_name)

    st.markdown("---")
    st.subheader("2. Įkelk JPG/PNG arba įklijuok Google paveikslėlio nuorodą")

    col_left, col_right = st.columns(2)

    with col_left:
        uploaded_file = st.file_uploader(
            "Pasirink failą",
            type=["jpg", "jpeg", "png"],
        )

    with col_right:
        url = st.text_input("Paveikslėlio URL (Copy image link iš Google)")

    img_to_use = None
    image_source_str = ""

    if uploaded_file is not None:
        img_to_use = Image.open(uploaded_file).convert("RGB")
        image_source_str = f"upload:{uploaded_file.name}"
        st.image(img_to_use, caption="🖼️ Įkeltas failas", use_column_width=True)

    elif url:
        try:
            resp = requests.get(url)
            img_to_use = Image.open(BytesIO(resp.content)).convert("RGB")
            image_source_str = f"url:{url}"
            st.image(img_to_use, caption="🖼️ Įkelta iš URL", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Nepavyko nuskaityti iš URL: {e}")

    st.markdown("---")
    st.subheader("3. Vienos nuotraukos spėjimas")

    if st.button("🔍 Gauti spėjimą iš įkeltos nuotraukos"):
        if img_to_use is None:
            st.warning("Pirmiausia įkelk failą arba įklijuok paveikslėlio URL.")
        elif model is None:
            st.error("❌ Modelis neįkeltas – patikrink .h5 failus.")
        else:
            x = preprocess_for_model(img_to_use, internal_model_name)
            preds = model.predict(x)[0]
            pred_class = int(np.argmax(preds))
            conf = float(np.max(preds))

            st.success(
                f"✅ Spėjimas ({internal_model_name}): "
                f"**{CLASS_LABELS[pred_class]}** (tikimybė ≈ {conf:.2f})"
            )

            # įrašom rezultatą į DB (true_class čia nežinoma)
            save_prediction_result(
                image_source=image_source_str or "upload",
                model_name=internal_model_name,
                true_class=None,
                predicted_class=pred_class,
                split="user",
            )

    st.markdown("---")
    st.subheader("4. Pridėti nuotrauką į treniravimo / testavimo rinkinį")

    col1, col2, col3 = st.columns(3)
    with col1:
        target_split = st.selectbox(
            "Kur dėti?",
            ["user_train", "user_test"],
        )
    with col2:
        class_name = st.selectbox("Tikroji ženklo klasė", CLASS_LABELS, index=0)
    with col3:
        add_btn = st.button("💾 Įrašyti į DB mokymui/testui")

    if add_btn:
        if img_to_use is None:
            st.warning("Pirmiausia įkelk paveikslėlį (failu arba URL).")
        else:
            class_id = CLASS_LABELS.index(class_name)
            save_user_image_to_db(img_to_use, class_id, target_split)
            st.success(
                f"💾 Nuotrauka išsaugota su klase **{class_name}**, split='{target_split}'."
            )

    st.markdown("---")
    st.subheader("5. Spėjimas be įkėlimo (atsitiktinė TEST nuotrauka iš DB)")

    if st.button("🎲 Parodyk atsitiktinį TEST ženklą ir spėjimą"):
        if model is None:
            st.error("❌ Modelis neįkeltas.")
        else:
            rec = get_random_test_record()
            if rec is None:
                st.error("DB nerasta TEST įrašų. Ar paleidai ingest_gtsrb_csv.py?")
            else:
                try:
                    img = load_image_from_record(rec)
                    st.image(
                        img,
                        caption=f"DB TEST paveikslas: {rec.filename} (tikroji klasė: {rec.class_id})",
                        use_column_width=True,
                    )
                except Exception as e:
                    st.error(f"❌ Nepavyko nuskaityti paveikslo: {e}")
                    img = None

                if img is not None:
                    x = preprocess_for_model(img, internal_model_name)
                    preds = model.predict(x)[0]
                    pred_class = int(np.argmax(preds))
                    conf = float(np.max(preds))

                    st.info(
                        f"🎯 Spėjimas ({internal_model_name}): "
                        f"**{CLASS_LABELS[pred_class]}** (tikimybė ≈ {conf:.2f}) | "
                        f"tikroji klasė: **{rec.class_id}**"
                    )

                    # įrašom rezultatą į DB
                    save_prediction_result(
                        image_source=rec.filename,
                        model_name=internal_model_name,
                        true_class=rec.class_id,
                        predicted_class=pred_class,
                        split=rec.split,
                    )

    st.markdown("---")
    st.subheader("6. Paleisti modelių treniravimą iš DB")

    col_cnn, col_mobilenet = st.columns(2)
    with col_cnn:
        if st.button("🏋️‍♂️ Pertreniruoti CNN (train_cnn.py)"):
            with st.spinner("Treniruoja CNN modelį..."):
                os.system("python src/train_cnn.py")
            st.success("✅ CNN treniravimas baigtas.")

    with col_mobilenet:
        if st.button("🏋️‍♂️ Pertreniruoti MobileNet (train_mobilenet_generator.py)"):
            with st.spinner("Treniruoja MobileNet modelį..."):
                os.system("python src/train_mobilenet_generator.py")
            st.success("✅ MobileNet treniravimas baigtas.")


# =========================================
# 2️⃣ Grafikai ir metrikos
# =========================================

with tab_graphs:
    st.subheader("1. CNN mokymo grafikai (accuracy / loss)")

    history_path = os.path.join("src", "cnn_history_simple.npz")
    if os.path.exists(history_path):
        data = np.load(history_path)
        acc = data["acc"]
        val_acc = data["val_acc"]
        loss = data["loss"]
        val_loss = data["val_loss"]
        epochs = range(1, len(acc) + 1)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.plot(epochs, acc, label="Train acc")
            ax.plot(epochs, val_acc, label="Val acc")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.plot(epochs, loss, label="Train loss")
            ax.plot(epochs, val_loss, label="Val loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("cnn_history_simple.npz nerastas – paleisk train_cnn.py, kad jį sugeneruotum.")

    st.markdown("---")
    st.subheader("2. Confusion matrix iš išsaugotų rezultatų (ModelResult)")

    db: Session = SessionLocal()
    try:
        # paimam visus rezultatus su žinoma tikra klase
        rows = (
            db.query(ModelResult)
            .filter(ModelResult.true_class != None)  # noqa: E711
            .all()
        )
    finally:
        db.close()

    if not rows:
        st.info("Kol kas nėra įrašytų rezultatų su true_class – padaryk kelis test spėjimus.")
    else:
        df = pd.DataFrame(
            [
                {
                    "true": r.true_class,
                    "pred": r.predicted_class,
                    "model": r.model_name,
                    "split": r.split,
                }
                for r in rows
            ]
        )

        model_choice = st.selectbox(
            "Pasirink modelį confusion matricai",
            sorted(df["model"].unique().tolist()),
        )

        df_model = df[df["model"] == model_choice]
        y_true = df_model["true"].values
        y_pred = df_model["pred"].values

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion matrix – {model_choice}")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        st.markdown("**Klasifikavimo ataskaita:**")
        report = classification_report(y_true, y_pred, zero_division=0)
        st.text(report)

    st.markdown("---")
    st.subheader("3. Klasės pasiskirstymas DB (GTSRBRecord)")

    db: Session = SessionLocal()
    try:
        counts = (
            db.query(GTSRBRecord.class_id, func.count(GTSRBRecord.id))
            .group_by(GTSRBRecord.class_id)
            .all()
        )
    finally:
        db.close()

    if counts:
        class_ids = [c[0] for c in counts]
        freqs = [c[1] for c in counts]
        labels = [
            CLASS_LABELS[i] if i < len(CLASS_LABELS) else str(i) for i in class_ids
        ]
        chart_df = pd.DataFrame({"Klasė": labels, "Kiekis": freqs})
        st.bar_chart(chart_df, x="Klasė", y="Kiekis", height=400)
    else:
        st.info("DB dar tuščia – pirmiausia suimportuok anotacijas ir/ar vartotojo nuotraukas.")


# =========================================
# 3️⃣ DB pavyzdžiai
# =========================================

with tab_db:
    st.subheader("Pavyzdžiai iš DB (TEST įrašai)")

    db: Session = SessionLocal()
    try:
        examples = (
            db.query(GTSRBRecord)
            .filter(GTSRBRecord.split == "test")
            .limit(12)
            .all()
        )
    finally:
        db.close()

    if not examples:
        st.info("DB nėra TEST įrašų. Ar paleidai ingest_gtsrb_csv.py?")
    else:
        cols = st.columns(4)
        for idx, rec in enumerate(examples):
            try:
                img = load_image_from_record(rec)
                caption = f"{rec.filename} | class={rec.class_id}"
                with cols[idx % 4]:
                    st.image(img, caption=caption, use_column_width=True)
            except Exception:
                with cols[idx % 4]:
                    st.write(f"Klaida: {rec.filename}")