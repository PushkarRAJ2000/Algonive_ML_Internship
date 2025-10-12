import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Path setup ===
base_path = os.path.dirname(os.path.abspath(__file__))
movies_path = os.path.join(base_path, "movie.csv")
ratings_path = os.path.join(base_path, "rating.csv")
tags_path = os.path.join(base_path, "tag.csv")
ratings_pkl_path = os.path.join(base_path, "ratings.pkl")


# Load Movies & Tags (small files) ===
movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)

# Load Ratings Smartly ===
if os.path.exists(ratings_pkl_path):
    ratings = pd.read_pickle(ratings_pkl_path)
else:
    # Load only 50k rows for now to keep system fast
    ratings = pd.read_csv(ratings_path, nrows=50000)
    
    # Save as pickle for faster next runs
    ratings.to_pickle(ratings_pkl_path)

movies['genres'] = movies['genres'].str.split('|')
all_genres = set(g for lst in movies['genres'] for g in lst)
for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)


# Display Samples ===
print("🎬 Movies Dataset (first 5 rows):")
print(movies.head(), "\n")

print("⭐ Ratings Dataset (first 5 rows):")
print(ratings.head(), "\n")

print("🏷️ Tags Dataset (first 5 rows):")
print(tags.head(), "\n")

# Shapes ===
print("📊 Dataset Shapes:")
print(f"Movies: {movies.shape}")
print(f"Ratings: {ratings.shape}")
print(f"Tags: {tags.shape}")

# Missing Values ===
print("\n⚠️ Missing Values:")
print(f"Movies:\n{movies.isnull().sum()}\n")
print(f"Ratings:\n{ratings.isnull().sum()}\n")
print(f"Tags:\n{tags.isnull().sum()}")

# EDA(Exploratory Data Analysis) Codes
# Ratings distribution
plt.figure(figsize=(8,5))
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Top 10 most rated movies
top_movies = ratings['movieId'].value_counts().head(10).index
top_movie_titles = movies[movies['movieId'].isin(top_movies)][['movieId','title']]
top_movie_counts = ratings['movieId'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_movie_counts.values, y=top_movie_titles['title'])
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie")
plt.show()

# Ratings per genre (subset for speed)
genre_cols = list(all_genres)
genre_ratings = movies[genre_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=genre_ratings.values, y=genre_ratings.index)
plt.title("Number of Movies per Genre")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create User-Movie Ratings Matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute Item-Item Similarity (movies)
item_similarity = cosine_similarity(user_movie_matrix.T)  # transpose to get movies as rows
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Function to get top N recommendations for a user
def recommend_movies(user_id, top_n=5):
    user_ratings = user_movie_matrix.loc[user_id]
    similar_scores = item_similarity_df.dot(user_ratings)
    # remove already rated movies
    already_rated = user_ratings[user_ratings > 0].index
    similar_scores = similar_scores.drop(already_rated)
    top_movies = similar_scores.sort_values(ascending=False).head(top_n)
    return movies[movies['movieId'].isin(top_movies.index)][['movieId','title']]

# Test recommendation for a sample user
sample_user_id = ratings['userId'].iloc[0]  # first user in dataset
print(f"\nTop 5 movie recommendations for User {sample_user_id}:")
print(recommend_movies(sample_user_id))

# Create genre matrix (user preference)
genre_cols = [c for c in movies.columns if c not in ['movieId','title','genres']]
user_genre_matrix = ratings.merge(movies[['movieId']+genre_cols], on='movieId')
user_genre_pref = user_genre_matrix.groupby('userId')[genre_cols].mean()

# Recommend function
def hybrid_recommend_movies(user_id, top_n=5, alpha=0.7):
    # Collaborative score
    user_ratings = user_movie_matrix.loc[user_id]
    collab_score = item_similarity_df.dot(user_ratings)
    
    # Content score
    user_pref = user_genre_pref.loc[user_id]
    movie_genres = movies.set_index('movieId')[genre_cols]
    content_score = movie_genres.dot(user_pref)
    
    # Hybrid score
    hybrid_score = alpha*collab_score + (1-alpha)*content_score
    # Remove already rated movies
    already_rated = user_ratings[user_ratings>0].index
    hybrid_score = hybrid_score.drop(already_rated, errors='ignore')
    # Top N movies
    top_movies = hybrid_score.sort_values(ascending=False).head(top_n)
    return movies[movies['movieId'].isin(top_movies.index)][['movieId','title']]

# Test hybrid recommendation
# Select small test user set (first 5 unique users)
test_users = ratings['userId'].unique()[:5]

for uid in test_users:
    top5 = hybrid_recommend_movies(uid, top_n=5)
    print(f"\nUser {uid} → Hybrid Top 5 Recommendations:")
    print(top5)

# Define test users (first 5 unique users)
test_users = ratings['userId'].unique()[:5]

def hybrid_precision_at_5(user_id):
    top_movies = hybrid_recommend_movies(user_id, top_n=5)['movieId']
    # movies user actually rated >=4
    liked_movies = ratings[(ratings['userId']==user_id) & (ratings['rating']>=4)]['movieId']
    if len(liked_movies) == 0:
        return None
    precision = len(set(top_movies) & set(liked_movies)) / 5
    return precision

# Calculate precision for each test user
for uid in test_users:
    p5 = hybrid_precision_at_5(uid)
    if p5 is not None:
        print(f"User {uid} → Hybrid Precision@5: {p5:.2f}")
    else:
        print(f"User {uid} → No highly rated movies to compare")

print(f"🎬 Total Movies: {len(movies)}")
print(f"⭐ Total Ratings: {len(ratings)}")
print(f"👥 Total Unique Users: {ratings['userId'].nunique()}")

avg_rating = ratings['rating'].mean()
print(f"📈 Average Rating: {avg_rating:.2f}")

print("\n🎭 Genre Distribution (Top 10):")
genre_distribution = movies.drop(columns=['movieId', 'title', 'genres']).sum().sort_values(ascending=False).head(10)
print(genre_distribution)

print("\nSample Recommendations (User 1):")
print(hybrid_recommend_movies(1).head(5))
