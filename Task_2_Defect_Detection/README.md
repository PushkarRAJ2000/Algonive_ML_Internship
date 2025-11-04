# 🧠 Task 2 — Defect Detection using Deep Learning  
### Algonive Machine Learning Internship (Oct 2025 – Jan 2026)
**Intern:** Pushkar Raj  
**Repository:** [Algonive_ML_Internship](https://github.com/PushkarRAJ2000/Algonive_ML_Internship)

---

## 📌 **Project Overview**
This project aims to develop a **Deep Learning-based Defect Detection System** for coating surfaces.  
The model classifies images of coated materials into **Defective** and **Non-Defective** categories using **Convolutional Neural Networks (CNN)**.

The focus was on building a **classification model** that can accurately detect coating surface anomalies to help automate quality inspection in manufacturing.

---

## 🗂️ **Dataset Used**
The dataset used for this project is publicly available:

📁 **Dataset Name:** CoatingVision: A Defect Dataset for Coating Manufacturing  
🔗 **Source:** [Figshare Dataset Link](https://figshare.com/articles/dataset/_b_CoatingVision_A_Defect_Dataset_for_Coating_Manufacturing_b_/29260121?file=55182476)

> **Note:** The dataset is **not uploaded to GitHub** due to size limitations (>600MB).  
> You can download it directly from the link above and place it in:
> ```
> Task_2_Defect_Detection/dataset/classification/
> ```

---

## ⚙️ **Project Structure**
Task_2_Defect_Detection/
│
├── dataset/
│ ├── classification/
│ │ ├── images/ # Downloaded dataset images
│ │ └── labels.csv # Image label file
│
├── prepare_dataset.py # Prepares and splits data
├── train_classification_model.py # CNN model training
├── evaluate_model.py # Evaluates and visualizes model performance
├── model/
│ └── train_classification_model.h5 # Saved trained model
└── README.md


## 🧪 **How to Run**
```bash
# Step 1: Install dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn

# Step 2: Prepare dataset
python prepare_dataset.py

# Step 3: Train the model
python train_classification_model.py

# Step 4: Evaluate model
python evaluate_model.py
