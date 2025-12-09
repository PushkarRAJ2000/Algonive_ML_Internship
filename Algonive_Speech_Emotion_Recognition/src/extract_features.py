import librosa
import numpy as np
import pandas as pd
from load_data import load_dataset

# Extract MFCC from 1 audio file
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Padding / truncating so all audios have same size
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    except Exception as e:
        print("Error processing:", file_path, e)
        return None


def build_feature_dataset():
    df = load_dataset()
    features = []
    labels = []

    for _, row in df.iterrows():
        mfcc = extract_mfcc(row['file_path'])

        if mfcc is not None:
            features.append(mfcc)
            labels.append(row['emotion'])

    features = np.array(features)
    labels = np.array(labels)

    print("Feature dataset shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Save for model training
    np.save(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/processed/features.npy", features)
    np.save(r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/processed/labels.npy", labels)


    print("Saved features & labels successfully.")


if __name__ == "__main__":
    build_feature_dataset()
