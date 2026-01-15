# 📱 Mobile Price Classification using Deep Learning

## 📌 Project Overview
This project builds a Deep Learning classification model using TensorFlow/Keras to predict smartphone price categories (0–3) based on technical specifications.

## 🎯 Objective
To classify mobile phones into four price ranges (low, medium, high, very high) using a neural network model.

## 🧠 Model Architecture
- Sequential Neural Network
- Dense Layers (128 → 64 → 4)
- ReLU & Softmax activation
- Dropout for regularization
- Optimizer: Adam

## 📊 Dataset
- Source: Kaggle – Mobile Price Classification
- Features: 20 numerical features
- Target: price_range (0–3)

## 🧪 Experiment
- Baseline model without Dropout
- Improved model with Dropout
- Final accuracy: **~95%**

## 📈 Evaluation
- Accuracy
- Confusion Matrix
- Classification Report

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Matplotlib / Seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
