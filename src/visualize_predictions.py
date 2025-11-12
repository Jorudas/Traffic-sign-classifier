
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# ✅ Įkeliam modelį
model = load_model("traffic_sign_cnn_new.h5")

# ✅ Kelių testinių ženklų keliai
test_images = [
    ("STOP ženklas (14)", "examples_gtsrb/class_14.ppm"),
    ("Ribotas greitis 50 (2)", "examples_gtsrb/class_2.ppm"),
    ("Kelias su teise pirmumo (12)", "examples_gtsrb/class_12.ppm"),
    ("Pėsčiųjų perėja (27)", "examples_gtsrb/class_27.ppm"),
    ("Vaikai kelyje (28)", "examples_gtsrb/class_28.ppm"),
]

# ✅ Kiekvieną ženklą prognozuojam
plt.figure(figsize=(12, 6))
for i, (label, path) in enumerate(test_images):
    if not os.path.exists(path):
        print(f"⚠️ Nerastas: {path}")
        continue

    img = cv2.imread(path)
    img_resized = cv2.resize(img, (32, 32))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title(f"{label}\nPrognozuota klasė: {predicted_class}")
    plt.axis("off")

plt.tight_layout()
plt.show()