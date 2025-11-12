
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Ženklų pavadinimai (pagal GTSRB klasių sąrašą)
CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles > 3.5 tons"
]

# ✅ 1. Įkeliame modelį
model = load_model("traffic_sign_cnn_weighted.h5")
print("✅ Modelis įkeltas sėkmingai!")

# ✅ 2. Įkelk savo nuotrauką (pvz. STOP ženklą)
img_path = "C:/1 JORUDAS/DOC Jorudas/AI/PROJEKTAI/traffic_sign_classifier/stop.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (32, 32))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array = np.expand_dims(img_rgb / 255.0, axis=0)

# ✅ 3. Prognozė
pred = model.predict(img_array)
pred_class = np.argmax(pred)
confidence = np.max(pred) * 100

# ✅ 4. Rezultatas
label = CLASSES[pred_class]
print(f"🧠 Atpažintas ženklas: {label}")
print(f"📊 Pasitikėjimas: {confidence:.2f}%")

# ✅ 5. Parodome rezultatą paveikslėlyje
plt.imshow(img_rgb)
plt.title(f"{label} ({confidence:.2f}%)", color="green")
plt.axis("off")
plt.show()