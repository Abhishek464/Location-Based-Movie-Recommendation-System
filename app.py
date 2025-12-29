import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import requests

# ---------------------------
# Download MovieLens dataset if not present
# ---------------------------
@st.cache_data
def download_and_load_data():
    if not os.path.exists("ml-1m"):
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = "ml-1m.zip"
        
        # Download dataset
        with open(zip_path, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        
        # Unzip dataset
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
    
    # Load data
    users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",encoding="latin-1",
                        names=["user_id","gender","age","occupation","zipcode"])
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["movie_id","title","genres"])
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",encoding="latin-1",
                          names=["user_id","movie_id","rating","timestamp"])
    
    ratings_users = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
    return ratings_users, users, movies, ratings

ratings_users, users, movies, ratings = download_and_load_data()

# ---------------------------
# Helper Functions
# ---------------------------
def get_global_top_movies(n=10):
    return (ratings_users.groupby("title")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n))

def get_location_top_movies(zipcode, n=10):
    local_ratings = ratings_users[ratings_users["zipcode"] == zipcode]
    if local_ratings.empty:
        return pd.Series([], dtype="float64")
    return (local_ratings.groupby("title")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n))

def get_collab_recommendations(zipcode, target_user=None, n=10):
    local_ratings = ratings_users[ratings_users["zipcode"] == zipcode]
    if local_ratings.empty:
        return []
    
    location_ratings = local_ratings.pivot_table(index="user_id",
                                                 columns="title",
                                                 values="rating").fillna(0)
    if location_ratings.shape[0] < 2:
        return []
    
    similarity_matrix = cosine_similarity(location_ratings)
    similarity_df = pd.DataFrame(similarity_matrix,
                                 index=location_ratings.index,
                                 columns=location_ratings.index)
    
    if not target_user:
        target_user = location_ratings.index[0]
    
    if target_user not in similarity_df.index:
        return []
    
    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:6]
    watched = set(location_ratings.loc[target_user][location_ratings.loc[target_user] > 0].index)
    
    recommend_scores = {}
    for sim_user, score in similar_users.items():
        movies_rated = location_ratings.loc[sim_user]
        for movie, rating in movies_rated.items():
            if movie not in watched and rating > 3:
                recommend_scores[movie] = recommend_scores.get(movie, 0) + score * rating
    
    recommendations = sorted(recommend_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return recommendations

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Location-Based Movie Recommendation System")

st.sidebar.header("User Settings")
zipcode = st.sidebar.text_input("Enter Zipcode", value="48067")
target_user = st.sidebar.number_input("Enter User ID (optional)", value=1, step=1)

st.write("## 🔹 Global Top Movies")
st.dataframe(get_global_top_movies())

st.write(f"## 🔹 Top Movies in Location `{zipcode}`")
location_top = get_location_top_movies(zipcode)
if location_top.empty:
    st.warning("No data for this location.")
else:
    st.dataframe(location_top)

st.write(f"## 🔹 Personalized Recommendations for User {target_user} in `{zipcode}`")
collab_recs = get_collab_recommendations(zipcode, target_user)
if not collab_recs:
    st.warning("Not enough data for collaborative recommendations.")
else:
    st.write(pd.DataFrame(collab_recs, columns=["Movie", "Score"]))
