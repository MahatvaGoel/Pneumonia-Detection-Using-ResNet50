import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = load_model("models/pneumonia_model.h5")

# =====================================
# LOAD IMAGE
# =====================================

image_path = "image.png"

img = cv2.imread(image_path)

# =====================================
# CHECK IMAGE
# =====================================

if img is None:
    print("Image not found!")
    exit()

# =====================================
# PREPROCESS IMAGE
# =====================================

img_resized = cv2.resize(img, (224,224))

img_normalized = img_resized / 255.0

img_input = np.expand_dims(img_normalized, axis=0)

# =====================================
# PREDICT
# =====================================

prediction = model.predict(img_input)[0][0]

# =====================================
# OUTPUT
# =====================================

print("\nPrediction Score:", prediction)

if prediction > 0.5:
    print("\nRESULT: PNEUMONIA DETECTED")
else:
    print("\nRESULT: NORMAL")