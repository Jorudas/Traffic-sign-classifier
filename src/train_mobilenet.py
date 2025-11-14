
# src/train_mobilenet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from data_loader import load_training_data
from preprocess import preprocess_image


print("📦 Krauname duomenis MobileNetV2 modeliui...")


# ==========================================
# 1) Nuskaitome VISUS duomenis iš DB
# load_training_data grąžina 4 reikšmes!
# ==========================================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_training_data(
    limit=None,
    target_size=(224, 224)
)

print(f"📊 X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")
print(f"📊 y_train_raw: {y_train_raw.shape}, y_test_raw: {y_test_raw.shape}")

# ==========================================
# 2) Sujungiame visus duomenis į vieną masyvą
# MobileNet mokysimės nuo NULIO
# (nors naudojame ImageNet svorius)
# ==========================================
X_raw = np.concatenate([X_train_raw, X_test_raw], axis=0)
y_raw = np.concatenate([y_train_raw, y_test_raw], axis=0)

print(f"📸 VISO vaizdų: {X_raw.shape}, Žymės: {y_raw.shape}")

NUM_CLASSES = len(np.unique(y_raw))
print(f"📚 Klasės: {NUM_CLASSES}")

# ==========================================
# 3) Daliname duomenis į train/validation
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
)

print("🔄 Pradėtas preprocess visiems vaizdams...")

# ==========================================
# 4) Taikome preprocess_image
# ==========================================
X_train = np.array([
    preprocess_image(img, target_size=(224, 224), normalize=True)
    for img in X_train
])

X_val = np.array([
    preprocess_image(img, target_size=(224, 224), normalize=True)
    for img in X_val
])

print("🧼 Normalizavimas ir ROI cropping pritaikyti.")

# ==========================================
# 5) Kuriame MobileNetV2 modelį
# ==========================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # 1 ETAPAS

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 1 ETAPAS: mokymas su užšaldyta baze...")

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ModelCheckpoint("mobilenet_stage1.h5", save_best_only=True, monitor="val_accuracy")
]

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 6) 2 ETAPAS: Fine-tuning — atrakiname bazę
# ==========================================
print("🔥 2 ETAPAS: Fine-tuning...")

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks2 = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("mobilenet_final_best.h5", save_best_only=True, monitor="val_accuracy")
]

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks2,
    verbose=1
)

print("🎉 VISKAS! Galutinis modelis: mobilenet_final_best.h5")