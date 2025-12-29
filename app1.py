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
    users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python", encoding='latin-1', # Added encoding
                        names=["user_id","gender","age","occupation","zipcode"])
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python",
                         names=["movie_id","title","genres"], encoding='latin-1')
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", encoding='latin-1', # Added encoding
                          names=["user_id","movie_id","rating","timestamp"])

    ratings_users = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
    return ratings_users, users, movies, ratings

ratings_users, users, movies, ratings = download_and_load_data()

# ---------------------------
# Helper Functions (unchanged core + added fallback)
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
    """
    Original collaborative recommendations based on users inside the zipcode.
    Returns list of tuples (movie_title, score)
    """
    local_ratings = ratings_users[ratings_users["zipcode"] == zipcode]
    if local_ratings.empty:
        return []  # no local data

    location_ratings = local_ratings.pivot_table(index="user_id",
                                                 columns="title",
                                                 values="rating").fillna(0)
    if location_ratings.shape[0] < 2:
        return []  # not enough users for CF

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
# New: zipcode-level pivot and similarity utilities for nearby-CF fallback
# ---------------------------

@st.cache_data
def build_zipcode_title_pivot():
    """
    Build and cache a zipcode x title pivot where values are mean rating (or 0 if missing).
    This is used to compute zipcode-to-zipcode similarity.
    """
    pivot = (ratings_users.groupby(['zipcode', 'title'])['rating']
             .mean()
             .unstack(fill_value=0))
    # pivot.index are zipcodes (strings); pivot.columns movie titles
    return pivot

def compute_zipcode_similarities(pivot):
    """
    Compute cosine similarity between zipcode vectors (rows of pivot).
    Returns a DataFrame similarity matrix (index and columns are zipcodes).
    """
    if pivot.shape[0] < 2:
        return pd.DataFrame()
    sim = cosine_similarity(pivot.values)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)
    return sim_df

def nearby_collab_fallback(zipcode, n_sim_zipcodes=5, top_n_items=10):
    """
    Fallback: if local CF fails, use zipcode-level collaborative filtering to find similar zipcodes
    and aggregate their top movies weighted by similarity.

    Returns list of tuples: (movie_title, aggregated_score, source_zipcode_list)
    """
    pivot = build_zipcode_title_pivot()
    # If pivot has no data, bail out
    if pivot.empty:
        return []

    # If the entered zipcode exists in pivot (has at least some movies), compute similarities and pick top similar zipcodes
    if zipcode in pivot.index:
        sim_df = compute_zipcode_similarities(pivot)
        if sim_df.empty:
            return []
        sims = sim_df.loc[zipcode].drop(labels=[zipcode], errors='ignore')
        top_zipcodes = sims.sort_values(ascending=False).head(n_sim_zipcodes)
        if top_zipcodes.empty:
            return []
        # Aggregate top movies from those zipcodes weighted by similarity
        weighted_scores = {}
        sources = {}  # keep track which zipcode contributed which movie (for debug/display)
        for other_zip, sim_score in top_zipcodes.items():
            # vector for other_zip is pivot.loc[other_zip] (movie -> mean rating)
            row = pivot.loc[other_zip]
            for movie, mean_rating in row[row > 0].items():
                weighted_scores[movie] = weighted_scores.get(movie, 0.0) + sim_score * mean_rating
                sources.setdefault(movie, set()).add(other_zip)
        if not weighted_scores:
            return []
        ranked = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_items]
        # attach list of source zipcodes for each movie
        return [(movie, score, ", ".join(sorted(sources.get(movie, [])))) for movie, score in ranked]
    else:
        # Entered zipcode has no history in pivot. Use heuristic nearby zipcodes:
        # 1) Try zipcodes with same first 3 digits (common locality grouping)
        prefix = str(zipcode)[:3]
        same_prefix = [z for z in pivot.index if str(z).startswith(prefix)]
        if same_prefix:
            # aggregate top movies from these zipcodes (unweighted)
            agg = {}
            sources = {}
            for z in same_prefix:
                row = pivot.loc[z]
                for movie, mean_rating in row[row > 0].items():
                    agg[movie] = agg.get(movie, 0.0) + mean_rating
                    sources.setdefault(movie, set()).add(z)
            if agg:
                ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n_items]
                return [(movie, score, ", ".join(sorted(sources.get(movie, [])))) for movie, score in ranked]
        # 2) Fallback: use zipcodes with the most user ratings (most data)
        counts = (ratings_users.groupby('zipcode').size()).sort_values(ascending=False)
        top_zipcodes_by_count = counts.index.tolist()[:n_sim_zipcodes]
        agg = {}
        sources = {}
        for z in top_zipcodes_by_count:
            if z not in pivot.index:
                continue
            row = pivot.loc[z]
            for movie, mean_rating in row[row > 0].items():
                agg[movie] = agg.get(movie, 0.0) + mean_rating
                sources.setdefault(movie, set()).add(z)
        if not agg:
            return []
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n_items]
        return [(movie, score, ", ".join(sorted(sources.get(movie, [])))) for movie, score in ranked]

# ---------------------------
# Top-level: get recommendations with fallback
# ---------------------------

def get_recommendations_with_nearby_fallback(zipcode, target_user=None, n=10):
    """
    Tries:
      1) direct local collaborative recommendations (per-user CF within zipcode)
      2) if none, zipcode-level CF fallback (nearby CF) using nearby_collab_fallback
    Returns (recs, source) where recs is list and source is 'local' or 'nearby' or 'none'
    """
    local_recs = get_collab_recommendations(zipcode, target_user, n=10) # Fixed: Changed n=n_sim_zipcodes to n=10
    if local_recs:
        # local_recs are (movie, score) pairs - normalize to consistent format (movie, score, source_zipcodes)
        return ([(m, s, zipcode) for m, s in local_recs], 'local')
    # else try nearby CF
    nearby = nearby_collab_fallback(zipcode, n_sim_zipcodes=5, top_n_items=n)
    if nearby:
        return (nearby, 'nearby')
    return ([], 'none')

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Location-Based Movie Recommendation System (with Nearby CF Fallback)")

st.sidebar.header("User Settings")
zipcode = st.sidebar.text_input("Enter Zipcode", value="55455")
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
recs, source = get_recommendations_with_nearby_fallback(zipcode, target_user, n=10)

if source == 'local':
    st.success("Showing collaborative recommendations based on users in the same zipcode.")
    df = pd.DataFrame(recs, columns=["Movie", "Score", "Source Zipcodes"])
    st.dataframe(df)
elif source == 'nearby':
    st.info("Not enough local data — showing recommendations aggregated from nearby / similar zipcodes.")
    df = pd.DataFrame(recs, columns=["Movie", "Aggregated Score", "Source Zipcodes"])
    st.dataframe(df)
else:
    st.warning("Not enough data for collaborative recommendations or fallback. Showing global top movies instead.")
    st.dataframe(get_global_top_movies())