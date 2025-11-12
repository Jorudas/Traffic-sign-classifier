
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import requests

# ✅ Įkeliam modelį
model = load_model("traffic_sign_cnn_new.h5")

# ✅ Veikiantis paveikslėlis iš Vikipedijos
IMAGE_URL = "https://share.google/images/1IIFgugvqFfTG0TcX"

# --- Atsisiunčiam ir išsaugom paveikslėlį ---
response = requests.get(IMAGE_URL)
open("real_stop.jpg", "wb").write(response.content)

# --- Nuskaitome ir paruošiame ---
img = cv2.imread("real_stop.jpg")

if img is None:
    raise FileNotFoundError("❌ Nepavyko įkelti paveikslėlio. Patikrink IMAGE_URL.")

img_resized = cv2.resize(img, (32, 32))
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# --- Prognozė ---
prediction = model.predict(img_input)
predicted_class = np.argmax(prediction)

# --- Rezultato atvaizdavimas ---
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Prognozuota klasė: {predicted_class}")
plt.axis("off")
plt.show()