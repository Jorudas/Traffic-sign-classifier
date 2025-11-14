
# src/app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

from labels import CLASS_LABELS   # ← Tavo LT pavadinimai

# =====================================================
# 1️⃣ Streamlit nustatymai
# =====================================================
st.set_page_config(page_title="GTSRB ženklų atpažinimas", layout="centered")
st.title("🚦 Kelių ženklų atpažinimo demo")
st.write("Įkelkite nuotrauką arba įklijuokite paveikslėlio nuorodą (Copy image link).")

# =====================================================
# 2️⃣ Įkeliame MobileNetV2 modelį
# =====================================================
@st.cache_resource
def load_mobilenet():
    model = load_model("mobilenet_final_best.h5")
    return model

model = load_mobilenet()
st.success("✅ MobileNetV2 modelis įkeltas!")


# =====================================================
# 3️⃣ Pasirinkimas – įkėlimas arba URL
# =====================================================
tab1, tab2 = st.tabs(["📁 Įkelti iš kompiuterio", "🌐 Įklijuoti nuorodą"])

uploaded_file = None
image_from_url = None
image = None

with tab1:
    uploaded_file = st.file_uploader("Pasirinkite kelio ženklo nuotrauką:", type=["jpg", "jpeg", "png"])

with tab2:
    url = st.text_input("Įklijuokite paveikslėlio nuorodą (Copy image link):")
    if url:
        try:
            response = requests.get(url)
            image_from_url = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Klaida įkeliant iš nuorodos: {e}")


# =====================================================
# 4️⃣ Paveikslėlio šaltinio pasirinkimas
# =====================================================
if uploaded_file:
    image = Image.open(uploaded_file)
elif image_from_url:
    image = image_from_url


# =====================================================
# 🔥 MobileNetV2 — paruošimas prognozei
# =====================================================
def prepare_mobilenet_image(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")

    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0   # normalizacija
    img = np.expand_dims(img, axis=0)               # (1, 224, 224, 3)
    return img


# =====================================================
# 5️⃣ Jei yra paveikslėlis — prognozuojam
# =====================================================
if image is not None:
    st.image(image, caption="Įkeltas ženklas", use_container_width=True)

    X = prepare_mobilenet_image(image)

    # Prognozė
    preds = model.predict(X)[0]
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # 🔥 Tikras LT pavadinimas
    label_name = CLASS_LABELS.get(pred_class, f"Klasė {pred_class}")

    # =====================================================
    # 🚦 Rezultato išvedimas
    # =====================================================
    st.markdown("### 🧠 Modelio prognozė:")
    st.write(f"**Klasė:** {pred_class} — {label_name}")
    st.write(f"**Tikimybė:** {confidence * 100:.2f}%")

    # Lentelė su TOP tikimybėmis
    probs_df = pd.DataFrame({
        "Klasė": list(range(len(preds))),
        "Tikimybė (%)": [round(p * 100, 2) for p in preds]
    })

    st.markdown("### 📊 TOP 10 klasės tikimybės:")
    st.dataframe(probs_df.sort_values("Tikimybė (%)", ascending=False).head(10))

else:
    st.info("👆 Įkelkite nuotrauką arba įklijuokite paveikslėlio nuorodą.")