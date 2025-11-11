
import cv2
import numpy as np
from keras.models import load_model
from src.labels import LABELS

# 🚀 Nurodome, kur yra išsaugotas modelis
MODEL_PATH = "traffic_sign_cnn.h5"

def predict_image(image_path):
    # ✅ 1. Įkeliame išsaugotą CNN modelį
    model = load_model(MODEL_PATH)

    # ✅ 2. Įkeliame nuotrauką iš disko
    img = cv2.imread(image_path)

    # ✅ Pakeičiame nuotraukos dydį į 32x32 (toks buvo modelio mokymas)
    img_resized = cv2.resize(img, (32, 32))

    # ✅ Konvertuojame į tinkamą formatą ir normalizuojame (nuo 0 iki 1)
    img_array = img_resized.astype("float32") / 255.0

    # ✅ Pridedame dimensiją (modelis laukia formos: 1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ 3. Darome prognozę
    prediction = model.predict(img_array)

    # ✅ Gauname didžiausią tikimybę turintį klasės ID
    class_id = np.argmax(prediction)

    # ✅ Iš klasės ID pasiimame pavadinimą (pvz. "Stop", "Yield", "50 km/h")
    class_name = LABELS.get(class_id, "Nežinoma klasė")

    print(f"✅ Atpažinta klasė: {class_id} → {class_name}")

    # ✅ 4. Užrašome rezultatą ant originalios nuotraukos
    cv2.putText(
        img,
        class_name,
        (10, 30),  # tekstas viršuje kairėje
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),  # žalia spalva
        2
    )

    # ✅ 5. Parodome nuotrauką lange
    cv2.imshow("AI atpažinimo rezultatas", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ✅ Ši dalis pasileidžia tik vykdant failą kaip programą
if __name__ == "__main__":
    test_img = "stop.jpg"  # čia įrašyk tikro failo pavadinimą, pvz. "stop.png"
    predict_image(test_img)