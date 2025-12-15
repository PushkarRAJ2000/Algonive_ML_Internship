# Algonive Machine Learning Internship

**Intern:** Pushkar Raj  
**Duration:** 10 Oct 2025 – 10 Jan 2026  
**Domain:** Machine Learning  

This repository contains all my tasks, projects, and learnings during my internship at Algonive.  
Each folder represents one task/project assigned during the internship.

📁 Project Submission Note:
This repository demonstrates practical, end-to-end machine learning projects completed during my internship at Algonive.

## 🧠 Tools & Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- VS Code / Google Colab



---

## 🧩 Task List

### Task 1 – Movie Recommendation System
[Go to Task 1 →](./Task_1_Movie_Recommendation_System)

**Description:**  
Developed a Movie Recommendation System using **Collaborative Filtering**, **Content-Based Filtering**, and **Hybrid approach**.  
The system generates personalized movie suggestions based on user ratings, preferences, and genres.  
Includes **data exploration, visualization, and evaluation** (Precision@5).

**Status:** ✅ Completed  

---

---

## 🧠 Task 2: Defect Detection (Computer Vision)
🔗 [Go to Task 2 →](./Task_2_Defect_Detection)

**Goal:** Build a Convolutional Neural Network (CNN) to detect defects like *Surface Crack, Delamination, and Pinhole* from manufacturing images.

**Tech Stack:** Python, TensorFlow, Keras, scikit-learn, Matplotlib, Pandas.

**Highlights:**
- Prepared and cleaned dataset using a custom data pipeline.
- Trained a CNN model achieving **~89% accuracy**.
- Evaluated performance using classification metrics and visualized sample predictions.
- Organized code modularly for dataset preparation, training, and evaluation.


### Task 3 – Algonive_Speech_Recognition_System
🗣️ Task 3: Speech Emotion Recognition (Audio ML)

🔗 [Go to Task 3 →](./Algonive_Speech_Emotion_Recognition)

**Goal:** Build a machine learning model that listens to an audio file and predicts the emotion such as Calm, Happy, Angry, Sad, Fearful, Neutral, Disgust, Surprise.

**Tech Stack:** Python, Librosa, NumPy, Pandas, Scikit-learn, Joblib.

**Highlights:**

Extracted MFCC audio features from the RAVDESS dataset.

Trained a Random Forest Classifier achieving ~71% accuracy.

Built a complete pipeline for data loading → feature extraction → model training → prediction.

Saved trained model (rf_model.pkl) for real-time emotion detection.

Added a clean README + GitHub project structure for easy evaluation.


### 🧠 Task 4 – Stock Price Prediction System

📈 Task 4: Stock Price Prediction (Time Series ML)

🔗[Go to Task 4 →](./Task_4_Stock_Prediction_System)

**Goal:**
Build a Stock Price Prediction System to forecast the next day’s closing price using historical stock market data.

**Dataset:**

Source: Kaggle (NSE India Historical Stock Data – ~197 MB)

Used a lightweight subset: WIPRO.NS.csv for efficient processing

**Tech Stack:**
Python, Pandas, NumPy, Matplotlib, scikit-learn, Joblib

**Highlights:**

Loaded and preprocessed historical NSE stock data

Visualized long-term stock price trends

Trained a Linear Regression model (lightweight & old-laptop friendly)

Achieved ~0.99 R² score on test data

Predicted next day’s stock closing price

Built a clean ML pipeline:
Data Loading → Visualization → Model Training → Prediction

Status: ✅ Completed
