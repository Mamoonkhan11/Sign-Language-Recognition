# Model trainer

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, optimizers
import os

from config.settings import PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH, LABELS_PATH


def train_mobilenet():

    # Ensure enough classes
    classes = [d for d in os.listdir(PROCESSED_DIR)
               if os.path.isdir(os.path.join(PROCESSED_DIR, d))]

    if len(classes) < 2:
        raise Exception(" Need at least TWO classes to train.")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        PROCESSED_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset="training"
    )

    val_gen = datagen.flow_from_directory(
        PROCESSED_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset="validation"
    )

    with open(LABELS_PATH, "w") as f:
        f.writelines("\n".join(train_gen.class_indices.keys()))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(train_gen.class_indices), activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    cb = [
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cb)