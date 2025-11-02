import pandas as pd
import os

# Step 1: Paths set karna
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "dataset", "classification")

csv_path = os.path.join(data_path, "labels.csv")
img_folder = os.path.join(data_path, "images")

# Step 2: CSV load karna
df = pd.read_csv(csv_path)
print("CSV loaded successfully!")
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# Step 3: Image path add karna
df["image_path"] = df["file_name"].apply(lambda x: os.path.join(img_folder, x))

# Step 4: Check missing images
missing = df[~df["image_path"].apply(os.path.exists)]
print(f"Missing images: {len(missing)}")

# Step 5: Summary
print("Total samples:", len(df))
print("Defect columns:", ["Surface_Crack", "Delamination", "Pinhole", "unclassified"])

# Step 6: Save as pickle for faster load later
pkl_path = os.path.join(base_path, "classification_data.pkl")
df.to_pickle(pkl_path)
print(f"Saved processed data to {pkl_path}")
