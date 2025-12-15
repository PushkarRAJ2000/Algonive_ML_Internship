📈 Task 4: Stock Price Prediction System
🔍 Overview

This project focuses on building a Stock Price Prediction System using historical stock market data.
The system predicts the next day’s closing price based on the previous day’s closing value using a lightweight Machine Learning model, making it suitable for low-resource systems.

This project is developed as part of the Algonive Machine Learning Internship.

🎯 Objective

Load and preprocess historical stock price data

Visualize stock price trends

Train a machine learning model for prediction

Predict the next day’s stock closing price

🧠 Model Used

Linear Regression

Chosen for:

Low computational cost

Fast training

Old-laptop friendly

Clear interpretability

📊 Dataset

Source: Kaggle

Dataset Name: NSE India Historical Stock Data

Total Size: ~197 MB

Link:
👉 https://www.kaggle.com/datasets/bhaktij/nse-stock-market-historical-data

🔹 Note: The full dataset contains multiple stocks.
For efficient processing, only WIPRO (WIPRO.NS.csv) was used in this project.

🗂 Project Structure
Task_4_Stock_Prediction_System/
│
├── data/
│   └── WIPRO.NS.csv
│
├── models/
│   └── linear_model.pkl
│
├── src/
│   ├── load_data.py
│   ├── visualize.py
│   ├── train_model.py
│   └── predict.py
│
└── README.md

⚙️ Workflow

Data Loading & Cleaning

Read CSV file

Convert date format

Sort time series data

Visualization

Line plot of stock closing price over time

Model Training

Feature: Previous day’s close price

Target: Current day’s close price

Train-test split without shuffling

Prediction

Predict next day’s closing price using trained model

📈 Results

Model Accuracy (R² Score): ~0.99

Prediction Output Example:

Last Close Price: 382.14
Predicted Next Day Price: 382.00

🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib

scikit-learn

Joblib

✅ Key Highlights

Lightweight & efficient implementation

Old-laptop friendly

Clean modular code

Real-world stock data

Fully reproducible pipeline

👤 Author

Pushkar Raj
Machine Learning Intern @ Algonive
