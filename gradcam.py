import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================================
# LOAD MODEL
# =========================================

model = load_model("models/pneumonia_model.h5")

# =========================================
# LOAD IMAGE
# =========================================

img_path = "image1.png"

img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

# =========================================
# PREPROCESS IMAGE
# =========================================

img = cv2.resize(img, (224,224))

original = img.copy()

img = img / 255.0

img = np.expand_dims(img, axis=0)

# =========================================
# GET RESNET MODEL
# =========================================

base_model = model.layers[0]

# =========================================
# LAST CONVOLUTION LAYER
# =========================================

last_conv_layer = base_model.get_layer("conv5_block3_out")

# =========================================
# CREATE GRADCAM MODEL
# =========================================

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

# =========================================
# COMPUTE GRADIENTS
# =========================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img)

    loss = tf.reduce_mean(predictions)

# =========================================
# GET GRADIENTS
# =========================================

grads = tape.gradient(loss, conv_outputs)

# =========================================
# GLOBAL AVERAGE POOLING
# =========================================

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

# =========================================
# FEATURE MAPS
# =========================================

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# =========================================
# NORMALIZE HEATMAP
# =========================================

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)

# =========================================
# CONVERT TO IMAGE
# =========================================

heatmap = cv2.resize(heatmap, (224,224))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# =========================================
# OVERLAY HEATMAP
# =========================================

superimposed_img = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)

# =========================================
# SHOW RESULTS
# =========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Original X-ray")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()