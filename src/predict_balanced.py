
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 🧠 Įkeliame subalansuotą modelį
print("📦 Įkeliame modelį iš traffic_sign_cnn_balanced.h5...")
model = load_model("traffic_sign_cnn_balanced.h5")
print("✅ Modelis įkeltas sėkmingai!\n")

# 🔢 Klasės (GTSRB ženklų pavadinimai)
classes = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles >3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles >3.5 tons"
]

# 📸 Nurodyk kelią iki nuotraukos (pvz. STOP ženklo)
image_path = "C:/1 JORUDAS/DOC Jorudas/AI/PROJEKTAI/traffic_sign_classifier/stop.jpg"

# 📥 Įkeliame ir apdorojame vaizdą
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (32, 32))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = img_rgb / 255.0
img_expanded = np.expand_dims(img_norm, axis=0)

# 🧠 Prognozė
pred = model.predict(img_expanded)
class_id = np.argmax(pred)
confidence = np.max(pred)

# 🏷️ Rezultatas
label = classes[class_id]
print(f"🧠 Atpažintas ženklas: {label}")
print(f"📊 Pasitikėjimas: {confidence * 100:.2f}%")

# 🖼️ Parodome vaizdą su rezultatu
plt.imshow(img_rgb)
plt.title(f"{label} ({confidence * 100:.2f}%)", color='green')
plt.axis("off")
plt.show()