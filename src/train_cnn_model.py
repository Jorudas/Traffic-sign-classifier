
# src/train_cnn_model.py
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import load_training_data, get_data_generators

# ============================================
# 1️⃣ Nuskaitom duomenis iš DB per data_loader
# ============================================
X_train, X_val, y_train, y_val = load_training_data(limit=5000, target_size=(64, 64))

# Klasės skaičius
NUM_CLASSES = len(np.unique(y_train))
print(f"📚 Klasės: {NUM_CLASSES}")

# Vienos karštos reikšmės
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)

# Sukuriam duomenų generatorius (su augmentacija)
train_flow, val_flow = get_data_generators(X_train, y_train_cat, X_val, y_val_cat, batch_size=64)

# ============================================
# 2️⃣ CNN modelio architektūra
# ============================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# ============================================
# 3️⃣ Kompiliuojam modelį
# ============================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# 4️⃣ Callbacks – kad sustotų kai pasiekia maksimumą
# ============================================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint("traffic_sign_cnn_best.h5", save_best_only=True, monitor='val_accuracy')
]

# ============================================
# 5️⃣ Treniruojam su augmentacija
# ============================================
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=15,
    verbose=1,
    callbacks=callbacks
)

# ============================================
# 6️⃣ Išsaugom galutinį modelį
# ============================================
model.save("traffic_sign_cnn_new.h5")
print("💾 Modelis išsaugotas kaip traffic_sign_cnn_new.h5")

# ============================================
# 7️⃣ Tikslumo santrauka
# ============================================
val_loss, val_acc = model.evaluate(val_flow, verbose=0)
print(f"✅ Galutinis tikslumas: {val_acc:.4f}")