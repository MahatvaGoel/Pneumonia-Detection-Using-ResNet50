# Pneumonia Detection Using ResNet50

## Overview

This project is an AI-powered Pneumonia Detection System built using Deep Learning and Transfer Learning techniques on Chest X-ray images.

The model uses ResNet50 architecture with TensorFlow and Keras for binary classification:

- NORMAL
- PNEUMONIA

The project also implements Grad-CAM Explainable AI to visualize infected lung regions.

---

## Features

- Chest X-ray Classification
- ResNet50 Transfer Learning
- Deep Learning-based Prediction
- Grad-CAM Heatmap Visualization
- AI Web App using Gradio
- Confidence Score Prediction

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Gradio
- ResNet50

---

## Project Structure

```bash
Pneumonia-Detection/
│
├── app.py
├── train.py
├── predict.py
├── gradcam.py
├── requirements.txt
├── models/
├── outputs/
├── screenshots/
```

---

## Model Accuracy

- Achieved high accuracy using Transfer Learning on Chest X-ray dataset.

---

## Grad-CAM Visualization

Grad-CAM is used to generate heatmaps highlighting infected lung regions used by the AI model for prediction.

---

## Dataset

Dataset Used:
Chest X-ray Images (Pneumonia)

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Prediction

```bash
python predict.py
```

### Generate Grad-CAM Heatmap

```bash
python gradcam.py
```

### Launch Web App

```bash
python app.py
```

---

## Author

Mahatva Goyal