
import cv2
import numpy as np
from tensorflow.keras.models import load_model

CLASS_NAMES = [str(i) for i in range(43)]  # 0–42 klasės

def predict_image(image_path):
    model = load_model("traffic_sign_cnn.h5")

    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"🟢 Prognozė: klasė {class_id}, pasitikėjimas {confidence:.2f}")
    return class_id

if __name__ == "__main__":
    # TESTAS — įkelk bet kokį JPG ar PPM failą
    predict_image("test.jpg")