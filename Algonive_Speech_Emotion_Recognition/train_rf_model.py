import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load features
features = np.load(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/processed/features.npy")
labels = np.load(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/processed/labels.npy")

print("Features Loaded:", features.shape)
print("Labels Loaded:", labels.shape)

# Flatten MFCC (40,174) → 6960 features
X = features.reshape(features.shape[0], -1)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)

print("Training model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Test Accuracy:", acc)

# Save model
joblib.dump(model, r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/models/rf_model.pkl")
joblib.dump(le, r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/models/label_encoder.pkl")


print("Model saved as rf_model.pkl")
