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
    users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",encoding='latin-1',
                        names=["user_id", "gender", "age", "occupation", "zipcode"])
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", encoding='latin-1',
                         names=["movie_id", "title", "genres"])
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", encoding='latin-1',
                          names=["user_id", "movie_id", "rating", "timestamp"])

    ratings_users = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
    # Ensure zipcode column is string (preserve leading zeros)
    ratings_users["zipcode"] = ratings_users["zipcode"].astype(str)
    users["zipcode"] = users["zipcode"].astype(str)
    return ratings_users, users, movies, ratings

ratings_users, users, movies, ratings = download_and_load_data()

# ---------------------------
# Nearby-zipcode helper
# ---------------------------
def generate_nearby_zip_candidates(zipcode_str, max_radius=5):
    """
    Yield zipcodes as strings in the order:
    0 -> original, then -1, +1, -2, +2, ...
    Preserves original string length with zfill if original was zero-padded.
    """
    try:
        orig_len = len(zipcode_str)
        base = int(zipcode_str)
    except ValueError:
        # If non-numeric zipcode, just return the original and nothing else
        yield zipcode_str
        return

    yield zipcode_str  # radius 0

    for r in range(1, max_radius + 1):
        for delta in (-r, r):
            candidate = base + delta
            if candidate < 0:
                continue
            yield str(candidate).zfill(orig_len)

def find_nearest_zip_with_data(zipcode_str, ratings_df, max_radius=5):
    """
    Returns (found_zipcode_str, distance) or (None, None) if nothing found.
    distance is integer absolute difference from original zipcode (0 means same).
    """
    try:
        orig_base = int(zipcode_str)
    except ValueError:
        # If non-numeric, just check exact match
        if zipcode_str in ratings_df["zipcode"].unique():
            return zipcode_str, 0
        return None, None

    checked = set()
    for candidate in generate_nearby_zip_candidates(zipcode_str, max_radius=max_radius):
        if candidate in checked:
            continue
        checked.add(candidate)
        local_ratings = ratings_df[ratings_df["zipcode"] == candidate]
        if not local_ratings.empty:
            distance = abs(int(candidate) - orig_base)
            return candidate, distance
    return None, None

# ---------------------------
# Helper Functions
# ---------------------------
def get_global_top_movies(n=10):
    return (ratings_users.groupby("title")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n))

def get_location_top_movies(zipcode, n=10, fallback_max_radius=5):
    # find nearest zipcode with data (including original)
    found_zip, dist = find_nearest_zip_with_data(zipcode, ratings_users, max_radius=fallback_max_radius)
    if found_zip is None:
        return pd.Series([], dtype="float64"), None  # no data found
    local_ratings = ratings_users[ratings_users["zipcode"] == found_zip]
    series = (local_ratings.groupby("title")["rating"]
              .mean()
              .sort_values(ascending=False)
              .head(n))
    return series, found_zip

def get_collab_recommendations(zipcode, target_user=None, n=10, fallback_max_radius=5):
    """
    Attempts collaborative recommendations for given zipcode.
    If zipcode has no data or not enough users, falls back to nearest zipcode within max radius.
    Returns (recommendations_list, used_zipcode) where recommendations_list is a list of (movie, score).
    """
    found_zip, dist = find_nearest_zip_with_data(zipcode, ratings_users, max_radius=fallback_max_radius)
    if found_zip is None:
        return [], None

    local_ratings = ratings_users[ratings_users["zipcode"] == found_zip]
    if local_ratings.empty:
        return [], found_zip

    location_ratings = local_ratings.pivot_table(index="user_id",
                                                 columns="title",
                                                 values="rating").fillna(0)
    # need at least 2 users with ratings for collaborative filtering
    if location_ratings.shape[0] < 2:
        return [], found_zip

    similarity_matrix = cosine_similarity(location_ratings)
    similarity_df = pd.DataFrame(similarity_matrix,
                                 index=location_ratings.index,
                                 columns=location_ratings.index)

    if not target_user:
        target_user = location_ratings.index[0]

    if target_user not in similarity_df.index:
        # if the specified target_user isn't present in this location, use the top user in that location
        target_user = location_ratings.index[0]

    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:6]
    watched = set(location_ratings.loc[target_user][location_ratings.loc[target_user] > 0].index)

    recommend_scores = {}
    for sim_user, score in similar_users.items():
        movies_rated = location_ratings.loc[sim_user]
        for movie, rating in movies_rated.items():
            if movie not in watched and rating > 3:
                recommend_scores[movie] = recommend_scores.get(movie, 0) + score * rating

    recommendations = sorted(recommend_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return recommendations, found_zip

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Location-Based Movie Recommendation System (nearby-pincode fallback)")

st.sidebar.header("User Settings")
zipcode = st.sidebar.text_input("Enter Zipcode", value="55455")
target_user = st.sidebar.number_input("Enter User ID (optional)", value=1, step=1)
fallback_radius = st.sidebar.slider("Nearby search radius (in zipcode steps)", min_value=1, max_value=10, value=3)

st.write("## 🔹 Global Top Movies")
st.dataframe(get_global_top_movies())

# Location top movies with fallback
st.write(f"## 🔹 Top Movies in Location `{zipcode}` (with nearby fallback up to ±{fallback_radius})")
location_top, used_zip_for_top = get_location_top_movies(zipcode, n=10, fallback_max_radius=fallback_radius)
if location_top.empty:
    st.warning(f"No data found for `{zipcode}` or nearby zipcodes within ±{fallback_radius}.")
else:
    if used_zip_for_top != zipcode:
        st.info(f"No data for `{zipcode}` — showing top movies for nearby zipcode `{used_zip_for_top}` instead.")
    st.dataframe(location_top)

# Collaborative recommendations with fallback
st.write(f"## 🔹 Personalized Recommendations for User {target_user} in `{zipcode}` (nearby fallback up to ±{fallback_radius})")
collab_recs, used_zip_for_collab = get_collab_recommendations(zipcode, target_user, n=10, fallback_max_radius=fallback_radius)
if used_zip_for_collab is None:
    st.warning(f"No location data available for `{zipcode}` or nearby zipcodes within ±{fallback_radius}.")
elif not collab_recs:
    if used_zip_for_collab != zipcode:
        st.warning(f"Not enough data for collaborative recommendations in `{zipcode}`. Checked nearby zipcode `{used_zip_for_collab}` but still not enough data for collaborative filtering.")
    else:
        st.warning("Not enough data for collaborative recommendations in this zipcode.")
else:
    if used_zip_for_collab != zipcode:
        st.info(f"No collaborative data for `{zipcode}` — using users from nearby zipcode `{used_zip_for_collab}` for recommendations.")
    st.write(pd.DataFrame(collab_recs, columns=["Movie", "Score"]))
