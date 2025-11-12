
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ✅ 1. Įkeliam išsaugotą CNN modelį
model = load_model("traffic_sign_cnn.h5")
print("✅ Modelis sėkmingai įkeltas!")

# ✅ 2. Nurodom STOP ženklo paveikslėlio kelią
image_path = "examples_gtsrb/class_14.ppm"

# ✅ 3. Užkraunam ir apdorojam paveikslėlį
img = cv2.imread(image_path)
img = cv2.resize(img, (32, 32))          # GTSRB modelis mokytas su 32x32 vaizdais
# Pabandome be spalvų konvertavimo (paliekam BGR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0                        # Normalizuojam reikšmes [0,1]
input_img = np.expand_dims(img, axis=0)  # Formuojam įėjimo formą (1, 32, 32, 3)

# ✅ 4. Atliekame prognozę
predictions = model.predict(input_img)
predicted_class = np.argmax(predictions)

print(f"🔍 Modelio prognozė: klasė {predicted_class}")

# ✅ 5. Parodome paveikslėlį ir rezultatą
plt.imshow(img)
plt.title(f"Prognozė: {predicted_class}")
plt.axis("off")
plt.show()