
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Image

# ============================================
# 🔹 1. Nustatymai
# ============================================
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 25

print("📦 Krauname duomenis iš DB per generatorių...")

# ============================================
# 🔹 2. Kuriame generatorių
# ============================================
def generate_batches(split="train", batch_size=BATCH_SIZE):
    db: Session = SessionLocal()
    images = db.query(Image).filter_by(split=split).all()

    total = len(images)
    print(f"✅ Rasta {total} įrašų duomenų bazėje ({split})")

    while True:
        for i in range(0, total, batch_size):
            batch_images = images[i:i + batch_size]
            X_batch, y_batch = [], []
            for img in batch_images:
                data = cv2.imread(img.path)
                if data is None:
                    continue
                data = cv2.resize(data, IMG_SIZE)
                X_batch.append(data)
                y_batch.append(img.class_id)

            X_batch = np.array(X_batch, dtype="float32") / 255.0
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

# ============================================
# 🔹 3. Skaičiuojam, kiek duomenų turim
# ============================================
db = SessionLocal()
train_total = db.query(Image).filter_by(split="train").count()
test_total = db.query(Image).filter_by(split="test").count()
print(f"📊 Mokymui: {train_total} | Testavimui: {test_total}")

# ============================================
# 🔹 4. CNN modelio kūrimas
# ============================================
print("🧠 Kuriame CNN modelį...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(43, activation='softmax')  # 43 GTSRB klasės
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================
# 🔹 5. Duomenų generatoriai (train/test)
# ============================================
train_gen = generate_batches("train", batch_size=BATCH_SIZE)
test_gen = generate_batches("test", batch_size=BATCH_SIZE)

# ============================================
# 🔹 6. Callback’ai (apsauga nuo pertreniravimo)
# ============================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("traffic_sign_cnn_full.keras", save_best_only=True)
]

# ============================================
# 🔹 7. Modelio mokymas
# ============================================
print("🚀 Pradedame pilną mokymą iš viso GTSRB rinkinio...")

steps_per_epoch = train_total // BATCH_SIZE
validation_steps = test_total // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================
# 🔹 8. Išsaugojimas
# ============================================
model.save("traffic_sign_cnn_full.keras")
print("💾 Modelis išsaugotas: traffic_sign_cnn_full.keras")
print("✅ Pilnas mokymas iš GTSRB baigtas!")