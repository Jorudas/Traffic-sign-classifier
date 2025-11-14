
# ============================================
# train_mobilenet_generator.py
# Profesionalus MobileNetV2 mokymas iš DB naudojant generatorius
# ============================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
from database import SessionLocal
from models import Image
from preprocess import preprocess_image


# ============================================
# 1. Nuskaitome duomenų bazę į DataFrame
# ============================================
def load_db_to_dataframe():
    db = SessionLocal()
    rows = db.query(Image).filter_by(split="train").all()
    db.close()

    paths = [r.path for r in rows]
    labels = [r.class_id for r in rows]

    df = pd.DataFrame({
        "filepath": paths,
        "label": labels
    })
    return df


print("📦 Krauname duomenis iš DB į DataFrame...")

df = load_db_to_dataframe()
print(f"📸 Vaizdų iš DB: {len(df)}")


# ============================================
# 2. Keras Sequence — generatorius RAM taupymui
# ============================================
class TrafficSignSequence(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X = []
        y = []

        for _, row in batch_df.iterrows():
            img = preprocess_image(row["filepath"], target_size=self.img_size)
            X.append(img)
            y.append(row["label"])

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================
# 3. Daliname į Train/Validation DataFrame
# ============================================
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True
)

print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}")


# ============================================
# 4. Sukuriame generatorius
# ============================================
train_gen = TrafficSignSequence(train_df, batch_size=32, img_size=(224, 224))
val_gen   = TrafficSignSequence(val_df, batch_size=32, img_size=(224, 224))


# ============================================
# 5. Sukuriam MobileNetV2 modelį
# ============================================
NUM_CLASSES = df["label"].nunique()
print(f"📚 Klasės: {NUM_CLASSES}")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # 1 etapas

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Pradedamas 1 etapas (bazinis MobileNet užšaldytas)...")

callbacks_1 = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ModelCheckpoint("mobilenet_stage1.h5", monitor="val_accuracy", save_best_only=True)
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks_1,
    verbose=1
)


# ============================================
# 6. 2 etapas — Fine Tuning
# ============================================

print("🔥 Pradedame 2 etapą — Fine Tuning...")

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_2 = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ModelCheckpoint("mobilenet_final_best.h5", monitor="val_accuracy", save_best_only=True)
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks_2,
    verbose=1
)

print("🎉 VISKAS! Modelis išsaugotas kaip mobilenet_final_best.h5")