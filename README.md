# 📱 Mobile Price Classification – Deep Learning Project

## 🔍 Project Overview
This project is a **Mobile Phone Price Classification System** built using **Deep Learning** with **TensorFlow & Keras**.

The model predicts the **price category of a smartphone** based on its hardware specifications, such as RAM, battery, camera, CPU, and screen features.

🎯 **Objective:**  
To classify smartphones into price categories using a data-driven and machine learning approach.

---

## 🧠 Problem Statement
Smartphone prices vary significantly based on their specifications.  
For non-experts, it is difficult to determine whether a phone belongs to a low, medium, or high price range just by looking at its specs.

This project aims to solve that problem by using a **Deep Learning model** to automatically classify smartphone prices.

---

## 💡 Solution
A **Neural Network model** was trained to perform **multi-class classification**, predicting one of the following categories:

- 💸 Low Price  
- 💰 Medium Price  
- 💎 High Price  
- 👑 Very High Price  

The model learns patterns from smartphone specifications and outputs the most likely price category.

---

## 📊 Dataset & Features
The dataset contains **20 numerical features** related to smartphone specifications, including:

- Battery Power (mAh)
- RAM (MB)
- Internal Memory (GB)
- CPU Clock Speed (GHz)
- Number of CPU Cores
- Front & Primary Camera (MP)
- Screen Height & Width
- Pixel Resolution (Height & Width)
- Device Weight & Thickness
- Connectivity Features (WiFi, Bluetooth, 3G, 4G, Dual SIM)
- Talk Time

The target variable is **price_range**, representing the price category.

---

## 🧠 Model & Approach
- **Model Type:** Fully Connected Neural Network (Dense Layers)
- **Framework:** TensorFlow & Keras
- **Preprocessing:** Feature scaling using StandardScaler
- **Problem Type:** Multi-class Classification
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam

---

## ✅ Model Performance
The model demonstrates strong and balanced performance across all price categories.

- **Accuracy:** ~95%
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Validation Tools:** Confusion Matrix & Classification Report

This indicates good generalization on unseen data.

---

## 🖥️ Application
A simple **Streamlit web application** was built to demonstrate the model.

### App Features:
- User-friendly input interface for phone specifications
- Instant price category prediction
- Prediction confidence visualization

This makes the model accessible even to non-technical users.

---

## 📁 Project Structure
mobile-price-classification/
│
├── app.py # Streamlit application
├── dataset/
│ └── train.csv # Training dataset
├── model/
│ └── final_mobile_price_model.keras
├── notebook/
│ └── training.ipynb # Model training & evaluation
├── README.md
└── requirements.txt

yaml
Copy code

---

## ⚙️ How to Run Locally
```bash
conda activate ml310
python -m streamlit run app.py
🛠️ Tech Stack
Python 3.10

TensorFlow / Keras

Scikit-learn

Pandas & NumPy

Streamlit

Matplotlib & Seaborn

📚 Key Learnings
Building deep learning models for classification problems

Feature preprocessing and scaling

Evaluating models using multiple metrics

Debugging real-world ML compatibility issues

Deploying ML models into user-facing applications

🚀 Future Improvements
Add model explainability (feature importance)

Improve UI/UX design

Add REST API for predictions

Experiment with other ML models (e.g., XGBoost)

👤 Author
Abdul Muin
AI & Machine Learning Enthusiast

This project was developed as part of a personal portfolio to demonstrate practical machine learning and application development skills.
