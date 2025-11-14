#Script: train_yes_no.py 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy("mixed_float16")

print(f"TensorFlow Version: {tf.__version__}")
print("GPU:", tf.config.list_physical_devices('GPU'))


BASE_PATH = r"F:\PATH TO DATA"
DATA_DIR = os.path.join(BASE_PATH, "characters_sorted")   

TFLITE_MODEL_PATH = os.path.join(BASE_PATH, "dyslexia_yes_no.tflite")
H5_MODEL_PATH = os.path.join(BASE_PATH, "dyslexia_yes_no.h5")

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 20

print(f"Loading data from {DATA_DIR} ...")

orig_train = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

orig_val = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

class_names = orig_train.class_names
print("Detected folders:", class_names)


REVERSAL_INDEX = class_names.index("Reversal")

def binary_label(x, y):
    return x, tf.where(y == REVERSAL_INDEX, 1, 0)

train_ds = orig_train.map(binary_label)
val_ds = orig_val.map(binary_label)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

print("Mapped classes: 1 = YES (Reversal), 0 = NO (Corrected + Normal)")


model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid", dtype="float32")   
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


model.save(H5_MODEL_PATH)
print(f"Saved .h5 model at: {H5_MODEL_PATH}")


y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\n--- Classification Report (0 = NO, 1 = YES) ---")
print(classification_report(y_true, y_pred, target_names=["NO", "YES"], zero_division=0))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NO", "YES"],
            yticklabels=["NO", "YES"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("yes_no_confusion_matrix.png")
plt.show()

print("Saved confusion matrix as yes_no_confusion_matrix.png")


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model at: {TFLITE_MODEL_PATH}")
print("YESNO Dyslexia Model Ready!")
