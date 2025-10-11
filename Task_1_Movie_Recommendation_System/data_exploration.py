#Exploring MovieLens Dataset

import pandas as pd

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")

# Display basic info
print("🎬 Movies Dataset:")
print(movies.head(), "\n")

print("⭐ Ratings Dataset:")
print(ratings.head(), "\n")

print("🏷️ Tags Dataset:")
print(tags.head(), "\n")

# Check dataset shapes
print("Movies Shape:", movies.shape)
print("Ratings Shape:", ratings.shape)
print("Tags Shape:", tags.shape)

# Check for missing values
print("\nMissing Values:")
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())
print("Tags:\n", tags.isnull().sum())
