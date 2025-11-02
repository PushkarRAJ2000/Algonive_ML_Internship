import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns

# ---- PATHS ----
base_dir = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_2_Defect_Detection\dataset\classification"
image_dir = os.path.join(base_dir, "images")
csv_path = os.path.join(base_dir, "labels.csv")
model_path = r"C:\Users\PUSHKAR\Documents\Algonive_ML_Internship\Task_2_Defect_Detection\classification_model.h5"

print("Looking for images in:", image_dir)

# ---- LOAD CSV ----
df = pd.read_csv(csv_path)
print("CSV loaded successfully! Shape:", df.shape)

# Check sample files
print("Sample image files:", df["file_name"].head().tolist())

# ---- IMAGE DATA GENERATOR ----
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col="file_name",
    y_col=["Surface_Crack", "Delamination", "Pinhole", "unclassified"],
    target_size=(128, 128),
    batch_size=32,
    class_mode="raw",   # multi-label
    shuffle=False
)

# ---- LOAD MODEL ----
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ---- EVALUATE ----
if len(test_gen) > 0:
    loss, acc = model.evaluate(test_gen)
    print(f"\n✅ Model evaluation completed!\nLoss: {loss:.4f}\nAccuracy: {acc:.4f}")
else:
    print("⚠️ No valid images found to evaluate. Please check paths and filenames.")

# ✅ Predictions
y_pred = model.predict(test_gen)
y_true = df[["Surface_Crack", "Delamination", "Pinhole", "unclassified"]].values

# ✅ Step 1: Threshold tuning for better defect detection
threshold = 0.4  # Slightly more sensitive than 0.5
y_pred_binary = (y_pred > threshold).astype(int)

print("\n✅ Classification Report (Tuned Threshold = 0.4):")
print(classification_report(y_true, y_pred_binary, target_names=["Surface_Crack", "Delamination", "Pinhole", "unclassified"]))

# ✅ Step 2: Confusion-style heatmap visualization
cm = multilabel_confusion_matrix(y_true, y_pred_binary)
labels = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]

plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    plt.subplot(2, 2, i + 1)
    sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
plt.tight_layout()
plt.show()

# ✅ Step 3: Visualize few predictions
plt.figure(figsize=(12, 8))
for i in range(5):
    idx = random.randint(0, len(test_gen.filenames) - 1)
    img_path = os.path.join(image_dir, test_gen.filenames[idx])
    img = plt.imread(img_path)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')

    true_label_idx = np.argmax(y_true[idx])
    pred_label_idx = np.argmax(y_pred_binary[idx])
    true_label = labels[true_label_idx]
    pred_label = labels[pred_label_idx]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"P:{pred_label}\nT:{true_label}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

# ✅ Step 4: Final summary
print("\n📊 FINAL PROJECT SUMMARY:")
print(f"✔ Total Images Evaluated: {len(test_gen.filenames)}")
print(f"✔ Model Accuracy: {acc:.4f}")
print(f"✔ Tuned Threshold: {threshold}")
print("✔ Observation: Model performs strongest on Surface_Crack; other defects need more balanced data.")
print("\n✅ Task-2 Defect Detection Evaluation Completed Successfully!")
