
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ✅ Įkeliam modelį
model = load_model("traffic_sign_cnn_new.h5")

# ✅ Įkeliam STOP ženklą (pvz. iš examples_gtsrb aplanko)
stop_path = "examples_gtsrb/class_14.ppm"  # STOP ženklas paprastai yra klasė 14
if not os.path.exists(stop_path):
    raise FileNotFoundError(f"STOP ženklas nerastas: {stop_path}")

# ✅ Apdorojam vaizdą
img = cv2.imread(stop_path)
img_resized = cv2.resize(img, (32, 32))
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# ✅ Prognozė
prediction = model.predict(img_input)
predicted_class = np.argmax(prediction)

print(f"🛑 Prognozuota klasė: {predicted_class}")