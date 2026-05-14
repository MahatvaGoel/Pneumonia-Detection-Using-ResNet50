import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================================
# LOAD MODEL
# =========================================

model = load_model("models/pneumonia_model.h5")

# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_xray(image):

    # Resize image
    img = cv2.resize(image, (224,224))

    # Normalize
    img = img / 255.0

    # Convert to batch
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    # Output
    if prediction > 0.5:

        confidence = prediction * 100

        return f"PNEUMONIA DETECTED\nConfidence: {confidence:.2f}%"

    else:

        confidence = (1 - prediction) * 100

        return f"NORMAL\nConfidence: {confidence:.2f}%"

# =========================================
# GRADIO INTERFACE
# =========================================

interface = gr.Interface(

    fn=predict_xray,

    inputs=gr.Image(),

    outputs="text",

    title="Pneumonia Detection from Chest X-rays",

    description="Upload a chest X-ray image to detect Pneumonia using ResNet50 AI Model"

)

# =========================================
# LAUNCH APP
# =========================================

interface.launch()