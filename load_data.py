import os
import pandas as pd

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

DATA_PATH = r"C:/Users/PUSHKAR/Documents/Algonive_ML_Internship/Algonive_Speech_Emotion_Recognition/data/raw/RAVDESS/"


def get_emotion(filename):
    parts = filename.split("-")
    emotion_code = parts[2]
    return emotion_map.get(emotion_code, "unknown")

def load_dataset():
    data_list = []

    for actor in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                emotion = get_emotion(file)
                file_path = os.path.join(actor_path, file)
                data_list.append([file_path, emotion])

    df = pd.DataFrame(data_list, columns=["file_path", "emotion"])
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print("Total audio files:", len(df))
