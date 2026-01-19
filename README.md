📱 Mobile Price Classification – Deep Learning Project
🔍 Project Overview

This project is a Mobile Phone Price Classification System built using Deep Learning (TensorFlow / Keras).
The model predicts the price category of a smartphone based on its hardware specifications.

🎯 Goal:
Help users estimate whether a phone belongs to:

Low Price

Medium Price

High Price

Very High Price

based on technical features such as RAM, battery, camera, CPU, and screen specs.

🧠 Machine Learning Approach

Problem Type: Multi-class classification

Model: Fully Connected Neural Network (Dense NN)

Framework: TensorFlow & Keras

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

The model was trained on a labeled dataset containing mobile phone specifications and price ranges.

📊 Model Performance

Test Accuracy: ~95%

Macro F1-Score: ~0.95

The model generalizes well across all 4 price categories.

Confusion Matrix and classification report were used to validate performance.

🧾 Features Used

The model uses 20 numerical features, including:

Battery Power (mAh)

RAM (MB)

Internal Memory (GB)

CPU Clock Speed (GHz)

Number of CPU Cores

Front & Primary Camera (MP)

Screen Height & Width

Pixel Resolution (Height & Width)

Device Weight & Thickness

Connectivity Features (WiFi, Bluetooth, 3G, 4G, Dual SIM)

Talk Time

🖥️ Application Interface

A simple Streamlit web application was built to:

Input phone specifications via sliders & inputs

Predict the price category

Display prediction confidence

The UI focuses on clarity and usability, making it easy for non-technical users.

📁 Project Structure
mobile-price-classification/
│
├── app.py                  # Streamlit application
├── dataset/
│   └── train.csv            # Training dataset
├── model/
│   └── final_mobile_price_model.keras
├── notebook/
│   └── training.ipynb       # Model training & evaluation
├── README.md
└── requirements.txt

⚙️ How to Run Locally
conda activate ml310
python -m streamlit run app.py

🛠️ Tech Stack

Python 3.10

TensorFlow / Keras

Scikit-learn

Pandas & NumPy

Streamlit

Matplotlib & Seaborn

💡 Key Learnings

Handling multi-class classification problems

Building and evaluating deep learning models

Feature scaling and preprocessing

Debugging real-world ML compatibility issues

Creating user-friendly ML applications

🚀 Future Improvements

Add model explainability (feature importance)

Improve UI design

Add REST API for predictions

Experiment with other models (XGBoost, Random Forest)

👤 Author

Abdul Muin
AI & Machine Learning Enthusiast

📌 This project was built as part of a personal portfolio to demonstrate practical machine learning and application development skills.
