import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import requests

# Optional mapping/geolocation deps (used only if installed / API key provided)
try:
    from streamlit_folium import st_folium
    import folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

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
            r.write(r.content)

        # Unzip dataset
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

    # Load data
    users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python", encoding='latin-1',
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
# Geocoding helpers (Google Maps API - optional)
# ---------------------------

def reverse_geocode_google(lat, lng, api_key):
    """
    Given lat,lng and a Google Maps Geocoding API key, return the postal_code if found.
    """
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
        r = requests.get(url)
        data = r.json()
        if data.get("status") != "OK":
            return None
        for res in data.get("results", []):
            for comp in res.get("address_components", []):
                if "postal_code" in comp.get("types", []):
                    return comp.get("long_name")
        return None
    except Exception:
        return None


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🎬 Location-Based Movie Recommendation System (GPS / Map + zipcode fallback)")

st.sidebar.header("User Settings")
zipcode = st.sidebar.text_input("Enter Zipcode (or leave empty and use the map)", value="")
target_user = st.sidebar.number_input("Enter User ID (optional)", value=1, step=1)
fallback_radius = st.sidebar.slider("Nearby search radius (in zipcode steps)", min_value=1, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("**Map / GPS options**")
use_map = st.sidebar.checkbox("Use map to pick my location (click on map)", value=False)

st.sidebar.markdown("If you want reverse geocoding (lat/lng -> postal code) to work automatically, provide a Google Maps Geocoding API key below. If not provided you can still pick a point on the map and manually type the postal code into the zipcode box.")
google_api_key = st.sidebar.text_input("Google Maps Geocoding API key (optional)", type="password")

# Show global top movies
st.write("## 🔹 Global Top Movies")
st.dataframe(get_global_top_movies())

picked_coords = None
if use_map:
    st.write("## 🗺️ Pick a location on the map")
    if not FOLIUM_AVAILABLE:
        st.warning("Optional packages 'folium' and 'streamlit_folium' are not installed.\nTo enable interactive map clicking, run: `pip install folium streamlit-folium` and restart the app.\nUntil then, you can still manually enter a zipcode in the sidebar.")
    else:
        # Create a folium map and let the user click to pick a location
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)  # default to India center; feel free to change
        st_map = st_folium(m, width=700, height=450, returned_objects=["last_clicked"])
        if st_map and st_map.get("last_clicked"):
            picked = st_map["last_clicked"]
            picked_coords = (picked.get("lat"), picked.get("lng"))
            st.success(f"Picked coords: {picked_coords}")
            st.map(data=pd.DataFrame([{"lat": picked_coords[0], "lon": picked_coords[1]}]))

# If map provided coords and API key is present, try reverse geocoding to a postal code
derived_zip = None
if picked_coords:
    if google_api_key:
        with st.spinner("Reverse-geocoding picked location to postal code using Google Geocoding API..."):
            derived_zip = reverse_geocode_google(picked_coords[0], picked_coords[1], google_api_key)
            if derived_zip:
                st.success(f"Derived postal code from location: {derived_zip}")
                zipcode = zipcode or derived_zip
            else:
                st.info("Could not derive postal code from coordinates. Please enter zipcode manually if you want location-specific recommendations.")
    else:
        st.info("No Google API key provided — unable to reverse-geocode. Please enter zipcode manually in the sidebar if you want location-specific recommendations.")

# Use zipcode (either from sidebar or derived)
effective_zip = zipcode if zipcode else None
if not effective_zip:
    st.warning("No zipcode selected. Global recommendations are shown above. Enter a zipcode in the sidebar or pick a location on the map and provide a Google API key for automatic postal-code lookup.")

# Location top movies with fallback
st.write(f"## 🔹 Top Movies in Location `{effective_zip or 'N/A'}` (with nearby fallback up to ±{fallback_radius})")
if effective_zip:
    location_top, used_zip_for_top = get_location_top_movies(effective_zip, n=10, fallback_max_radius=fallback_radius)
    if location_top.empty:
        st.warning(f"No data found for `{effective_zip}` or nearby zipcodes within ±{fallback_radius}.")
    else:
        if used_zip_for_top != effective_zip:
            st.info(f"No data for `{effective_zip}` — showing top movies for nearby zipcode `{used_zip_for_top}` instead.")
        st.dataframe(location_top)
else:
    st.info("No zipcode provided — cannot compute location-specific top movies.")

# Collaborative recommendations with fallback
st.write(f"## 🔹 Personalized Recommendations for User {target_user} in `{effective_zip or 'N/A'}` (nearby fallback up to ±{fallback_radius})")
if effective_zip:
    collab_recs, used_zip_for_collab = get_collab_recommendations(effective_zip, target_user, n=10, fallback_max_radius=fallback_radius)
    if used_zip_for_collab is None:
        st.warning(f"No location data available for `{effective_zip}` or nearby zipcodes within ±{fallback_radius}.")
    elif not collab_recs:
        if used_zip_for_collab != effective_zip:
            st.warning(f"Not enough data for collaborative recommendations in `{effective_zip}`. Checked nearby zipcode `{used_zip_for_collab}` but still not enough data for collaborative filtering.")
        else:
            st.warning("Not enough data for collaborative recommendations in this zipcode.")
    else:
        if used_zip_for_collab != effective_zip:
            st.info(f"No collaborative data for `{effective_zip}` — using users from nearby zipcode `{used_zip_for_collab}` for recommendations.")
        st.write(pd.DataFrame(collab_recs, columns=["Movie", "Score"]))
else:
    st.info("No zipcode provided — cannot compute personalized recommendations.")

# Helpful notes and instructions
st.markdown("---")
st.markdown("**Notes & Setup**:\n\n- To enable the interactive map to pick a location, install the optional packages: `pip install folium streamlit-folium`.\n- If you want automatic reverse-geocoding (lat/lng -> postal code), supply a Google Maps Geocoding API key in the sidebar.\n  Example Google reverse geocode endpoint used in the code: `https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={API_KEY}`.\n- If you prefer Google Maps for display (instead of Leaflet/folium), you can replace the folium map in the code with a Google Maps embed or the Google Maps JS API, but that requires a Maps API key and a different embedding approach.\n- The zipcode-based recommendation logic and nearby-zipcode fallback (±k steps) use the MovieLens user `zipcode` field and do not rely on external geocoding unless you choose to enable it with the API key.")
