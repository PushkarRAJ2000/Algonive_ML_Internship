import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Paths
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "dataset/classification/labels.csv")
image_dir = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_2_Defect_Detection\dataset\classification\images"
# Load CSV
df = pd.read_csv(csv_path)
print("CSV loaded successfully!")
print(f"Columns: {df.columns.tolist()}")
print(f"Total samples: {len(df)}")

# Keep only existing images
df['filepath'] = df['file_name'].apply(lambda x: os.path.join(image_dir, x))
df = df[df['filepath'].apply(os.path.exists)]
print(f"Valid images: {len(df)}")

# Features and labels
label_cols = ['Surface_Crack', 'Delamination', 'Pinhole', 'unclassified']

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Image Data Generator
train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col=label_cols,
    target_size=(128,128),
    class_mode='raw',
    batch_size=32,
    shuffle=True
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col=label_cols,
    target_size=(128,128),
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_cols), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(train_gen, validation_data=test_gen, epochs=3)

# Save model
model.save(os.path.join(base_path, "classification_model.h5"))
print("Model trained and saved successfully!")
