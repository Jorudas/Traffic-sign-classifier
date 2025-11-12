
# src/app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from PIL import Image
import requests
from io import BytesIO

# =====================================================
# 1️⃣ Streamlit nustatymai
# =====================================================
st.set_page_config(page_title="GTSRB ženklų atpažinimas", layout="centered")
st.title("🚦 Kelių ženklų atpažinimo demo")
st.write("Įkelkite nuotrauką iš kompiuterio arba įklijuokite paveikslėlio nuorodą (pvz. iš Google).")

# =====================================================
# 2️⃣ Įkeliame modelį
# =====================================================
@st.cache_resource
def load_cnn_model():
    model = load_model("traffic_sign_cnn_new.h5")
    return model

model = load_cnn_model()
st.success("✅ Modelis įkeltas sėkmingai!")

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

# Pasirenkam, kuris šaltinis buvo panaudotas
if uploaded_file:
    image = Image.open(uploaded_file)
elif image_from_url:
    image = image_from_url

# =====================================================
# 4️⃣ Jei yra paveikslėlis — prognozuojam
# =====================================================
if image is not None:
    st.image(image, caption="Įkeltas ženklas", use_container_width=True)

    # Konvertuojam į numpy ir apdorojam
    image_np = np.array(image)
    try:
        processed = preprocess_image(image_np, target_size=(64, 64))
    except Exception as e:
        st.error(f"Klaida apdorojant paveikslėlį: {e}")
        st.stop()

    X = np.expand_dims(processed, axis=0)

    # Prognozė
    pred = model.predict(X)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # =====================================================
    # 5️⃣ Rezultato išvedimas
    # =====================================================
    st.markdown("### 🧠 Modelio prognozė:")
    st.write(f"**Klasė:** {pred_class}")
    st.write(f"**Tikimybė:** {confidence*100:.2f}%")

    # Lentelės vaizdas su visomis tikimybėmis
    probs_df = pd.DataFrame({
        "Klasė": list(range(len(pred[0]))),
        "Tikimybė (%)": [round(p*100, 2) for p in pred[0]]
    })
    st.dataframe(probs_df.sort_values("Tikimybė (%)", ascending=False).head(10))

else:
    st.info("👆 Įkelkite nuotrauką arba įklijuokite paveikslėlio nuorodą.")