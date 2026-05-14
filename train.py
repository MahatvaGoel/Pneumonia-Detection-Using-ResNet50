import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/val"
TEST_PATH = "dataset/test"

# ==========================================
# DATA PREPROCESSING + AUGMENTATION
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================
# LOAD DATASETS
# ==========================================

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ==========================================
# LOAD PRETRAINED RESNET50
# ==========================================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# ==========================================
# FREEZE MOST LAYERS
# ==========================================

for layer in base_model.layers[:-20]:
    layer.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')

])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.000001
)

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# ==========================================
# SAVE MODEL
# ==========================================

if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/pneumonia_model.h5")

print("\nModel trained and saved successfully!")

# ==========================================
# EVALUATE MODEL
# ==========================================

loss, accuracy = model.evaluate(test_data)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(test_data)

predicted_classes = (predictions > 0.5).astype("int32")

true_classes = test_data.classes

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=['NORMAL', 'PNEUMONIA']
))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:\n")
print(cm)

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train Accuracy', 'Validation Accuracy'])

plt.show()

# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8,6))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Train Loss', 'Validation Loss'])

plt.show()