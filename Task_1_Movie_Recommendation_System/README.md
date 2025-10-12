# 🎬 Movie Recommendation System

**Internship:** Machine Learning @ [Algonive](https://www.algonive.com/)  
**Intern:** Pushkar Raj  
**Task:** 1 – Movie Recommendation System  

---

## 📖 Project Overview
This project builds a **Movie Recommendation System** that suggests films based on **user preferences**, **ratings**, and **viewing history**.  
It combines **Collaborative Filtering**, **Content-Based Filtering**, and a **Hybrid approach** to generate personalized recommendations.

**Key Features:**
- **User-Based Recommendations:** Suggest movies based on similar users’ preferences.  
- **Content-Based Filtering:** Recommend movies with similar genres.  
- **Hybrid Recommendations:** Combine collaborative and content-based filtering for better accuracy.  
- **Ratings & Reviews Analysis:** Optional sentiment analysis for refining recommendations.  
- **Data Exploration & Visualization:** Insights into genres, ratings, and user behavior.

---

## 🧰 Tools & Technologies
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment:** VS Code / Google Colab  

---

## 📂 Dataset
- **MovieLens Dataset** (https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)  
- **Movies:** 27,278 entries  
- **Ratings:** 50,000 sample ratings (subset of full dataset)  
- **Tags:** 465,564 entries  
- Contains **movieId, title, genres, userId, rating, timestamp, tags**

---

## 📊 Sample Output
**Top 5 Movie Recommendations for User 1:**
| movieId | Title                        |
|---------|-------------------------------|
| 1270    | Back to the Future (1985)    |
| 2571    | Matrix, The (1999)           |
| 2916    | Total Recall (1990)          |
| 2985    | RoboCop (1987)               |
| 3448    | Good Morning, Vietnam (1987) |

**Genre Distribution (Top 10):**
Drama 13344
Comedy 8374
Thriller 4178
Romance 4127
Action 3520
Crime 2939
Horror 2611
Documentary 2471
Adventure 2329
Sci-Fi 1743


---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/PushkarRAJ2000/Algonive_ML_Internship.git

# Navigate to Task 1 folder
cd Algonive_ML_Internship/Task_1_Movie_Recommendation_System/data

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the script
python data_exploration.py

✅ Status

Data Exploration & Cleaning ✅

User-Based Recommendations ✅

Content-Based Recommendations ✅

Hybrid Recommendations ✅

Evaluation Metrics (Precision@5) ✅

💬 Acknowledgement
Thanks to @Algonive for providing this internship and guidance to explore real-world Machine Learning applications.
