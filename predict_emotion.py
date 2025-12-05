import numpy as np
import joblib
import librosa

# Load saved model & label encoder
model = joblib.load(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/models/rf_model.pkl")
label_encoder = joblib.load(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/models/label_encoder.pkl")

# MFCC extractor (same as training)
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Padding/truncate for consistent shape
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.reshape(1, -1)  # flatten for RF model

# Predict emotion
def predict(file_path):
    mfcc = extract_mfcc(file_path)
    pred = model.predict(mfcc)
    emotion = label_encoder.inverse_transform(pred)[0]
    return emotion

if __name__ == "__main__":
    # Test with an audio file
    test_file = test_file = r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/raw/RAVDESS/Actor_01/03-02-06-02-02-02-01.wav"
  # change path
    print("Predicted Emotion:", predict(test_file))
