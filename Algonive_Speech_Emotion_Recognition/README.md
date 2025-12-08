This is Task-3 of my Machine Learning Internship at Algonive.
In this project, I built a Speech Emotion Recognition System using MFCC audio features and a Random Forest model.

🎯 Objective
Predict the emotion expressed in an audio speech file using classical Machine Learning.

🛠️ Tech Stack
Python
Librosa (for audio feature extraction)
NumPy / Pandas
Scikit-Learn
Joblib

📂 Project Structure
Task_3/
│
├── src/
│   ├── load_data.py
│   ├── extract_features.py
│   ├── train_rf_model.py
│   └── predict_emotion.py
│
├── models/
│   ├── rf_model.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── processed/   (features not uploaded – size too large)
│
└── README.md

🎵 Dataset
Used RAVDESS Speech Emotion Dataset
(Link publicly available online)

🔍 Model Used
Random Forest Classifier
Achieved around 71% accuracy

📈 Pipeline
Load raw audio files
Extract MFCC features
Train Random Forest model
Save model + label encoder

Predict emotion from a test audio file

▶️ Run Prediction
python predict_emotion.py

🙌 Thanks

Thanks to Algonive for this Machine Learning internship opportunity!
