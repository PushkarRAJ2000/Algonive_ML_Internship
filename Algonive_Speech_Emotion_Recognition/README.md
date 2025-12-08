# 🎤 Speech Emotion Recognition — Task 3 (Algonive ML Internship)

This is **Task-3** of my **Machine Learning Internship at Algonive**.  
In this project, I built a **Speech Emotion Recognition System** using MFCC audio features and a Random Forest model.

---

## 🎯 Objective
Predict the **emotion** expressed in an audio speech file using classical Machine Learning.

---

## 🛠️ Tech Stack
- Python  
- Librosa (for audio feature extraction)  
- NumPy  
- Pandas  
- Scikit-Learn  
- Joblib  
- SoundFile / AudioRead  

---

## 📂 Project Structure

Task_3/
│
├── src/
│ ├── load_data.py
│ ├── extract_features.py
│ ├── train_rf_model.py
│ └── predict_emotion.py
│
├── models/
│ ├── rf_model.pkl
│ └── label_encoder.pkl
│
├── data/
│ └── processed/ (not uploaded due to size limit)
│
└── README.md


---

## 🎵 Dataset

**RAVDESS Speech Emotion Audio Dataset**  
Download Link: https://zenodo.org/record/1188976

(Contains emotional speech from 24 actors with 8 emotion classes)

---

## 🔍 Model Details
- Model: **Random Forest Classifier**
- Features: **MFCC (40 coefficients)**
- Accuracy: **~71%**

---

## 📈 Pipeline Overview

1. Load raw audio files  
2. Extract MFCC features using Librosa  
3. Create training dataset  
4. Train Random Forest classifier  
5. Save model and label encoder  
6. Predict emotion from audio file  

---

## ▶️ How to Run Prediction

### 1) Install dependencies  
pip install librosa numpy pandas scikit-learn joblib soundfile audioread

### 2) Run prediction script  
python predict_emotion.py

---

## 📌 Output Example

Predicted Emotion: calm


---

## 🙌 Acknowledgement

Thanks to **Algonive** for providing this Machine Learning internship opportunity  
and guiding me through real-world ML projects.

