
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
from labels import LABELS as labels  # importuojame ženklų pavadinimus

# ============================================================
# 🔹 ĮKELIAME IŠSAUGOTĄ MODELĮ
# ============================================================
MODEL_PATH = "traffic_sign_cnn.h5"
print(f"📦 Įkeliame modelį iš {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("✅ Modelis įkeltas sėkmingai!\n")

# ============================================================
# 🔹 FAILO PASIRINKIMAS PER NARŠYKLĘ
# ============================================================
Tk().withdraw()  # paslepia pagrindinį Tk langą
image_path = filedialog.askopenfilename(
    title="Pasirinkite nuotrauką",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("⚠️ Nepasirinkta jokia nuotrauka. Programa nutraukiama.")
    exit()

print(f"🖼️ Pasirinkta nuotrauka: {image_path}")

# ============================================================
# 🔹 VAIZDO APDOROJIMAS
# ============================================================
img = cv2.imread(image_path)
if img is None:
    print("❌ Klaida: Nepavyko nuskaityti paveikslėlio.")
    exit()

img_resized = cv2.resize(img, (32, 32))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# ============================================================
# 🔹 PROGNOZĖ
# ============================================================
predictions = model.predict(img_input)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions) * 100

# ============================================================
# 🔹 REZULTATO IŠVESTIS
# ============================================================
label_name = labels[predicted_class] if predicted_class < len(labels) else "Nežinomas ženklas"

print("\n🧠 Atpažinimo rezultatas:")
print(f"Ženklas: {label_name}")
print(f"Pasitikėjimas: {confidence:.2f}%")

# ============================================================
# 🔹 RODOME NUOTRAUKĄ SU ATPAŽINIMO TEKSTU
# ============================================================
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f"{label_name} ({confidence:.1f}%)", (10, 30), font, 0.8, (0, 255, 0), 2)
cv2.imshow("Atpažintas ženklas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()