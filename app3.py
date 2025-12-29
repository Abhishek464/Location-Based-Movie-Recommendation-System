import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import requests
import streamlit.components.v1 as components

# Optional geocoding dependency (reverse geocode lat/lon -> postcode)
# If not installed, the app will show instructions to install geopy or ask user for zipcode.
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

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
# Nearby-zipcode helper (unchanged)
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
# Helper Functions (unchanged)
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
# New: Reverse-geocode coords -> zipcode
# ---------------------------
def coords_to_zipcode(lat, lon, user_agent="movie_recommender_app"):
    """
    Uses geopy.Nominatim to reverse-geocode latitude/longitude to a postal code (zipcode).
    Returns zipcode as string, or None if not found or geopy unavailable.
    NOTE: Nominatim has rate limits - we use a rate-limiter if available.
    """
    if not GEOPY_AVAILABLE:
        return None

    try:
        geolocator = Nominatim(user_agent=user_agent)
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=2)
        loc = reverse((lat, lon), language="en", exactly_one=True, addressdetails=True)
        if not loc:
            return None
        ad = loc.raw.get("address", {})
        # Try common fields where postal code might appear
        for key in ("postcode", "postal_code", "zip"):
            if key in ad:
                return str(ad[key])
        # Sometimes postcode can be inside state/district strings - but that's rare.
        return None
    except Exception:
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("🎬 Location-Based Movie Recommendation System (GPS + nearby-pincode fallback)")

st.sidebar.header("User Settings")

# Option: user can either enter zipcode directly, or provide GPS coordinates (lat/lon)
use_gps = st.sidebar.checkbox("Use GPS coordinates (lat / lon) instead of entering zipcode", value=False)

lat_input = None
lon_input = None
inferred_zip_from_coords = None

if use_gps:
    st.sidebar.write("Provide coordinates below. If you prefer, click the 'Get browser GPS' helper to show your coordinates and copy-paste them into the inputs.")
    # Manual lat/lon inputs
    lat_input = st.sidebar.text_input("Latitude (e.g. 40.7128)", "")
    lon_input = st.sidebar.text_input("Longitude (e.g. -74.0060)", "")

    # Small browser-GPS helper: shows a tiny widget that gets coords from navigator.geolocation
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick helper:** click the button and allow location access in your browser. If the app host blocks automatic communication, the coordinates will display below — copy them into the lat/lon fields.")
    gps_html = """
    <div id="gps-box">
      <button onclick="getGPS()" style="padding:6px 8px;">Get browser GPS coordinates</button>
      <div id="coords" style="margin-top:6px; font-size:small;color:#333;"></div>
    </div>
    <script>
    async function getGPS(){
      const el = document.getElementById('coords');
      if(!navigator.geolocation){
        el.textContent = 'Geolocation not supported by your browser.';
        return;
      }
      el.textContent = 'Requesting location…';
      navigator.geolocation.getCurrentPosition(function(pos){
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        el.innerHTML = 'Lat: <b>'+lat+'</b><br>Lon: <b>'+lon+'</b><br><br><i>Copy these values into the latitude & longitude fields in the sidebar.</i>';
      }, function(err){
        el.textContent = 'Error getting location: ' + err.message;
      }, {enableHighAccuracy:true, timeout:10000});
    }
    </script>
    """
    # render small HTML helper (height tuned)
    #st.sidebar.components.v1.html(gps_html, height=140)
    components.html(gps_html, height=140)


    # If user provided coords, try to reverse-geocode to zipcode
    if lat_input.strip() != "" and lon_input.strip() != "":
        try:
            lat_val = float(lat_input.strip())
            lon_val = float(lon_input.strip())
            if GEOPY_AVAILABLE:
                with st.spinner("Reverse-geocoding coordinates to postal code..."):
                    zipcode_from_coords = coords_to_zipcode(lat_val, lon_val)
                if zipcode_from_coords:
                    inferred_zip_from_coords = zipcode_from_coords
                    st.sidebar.success(f"Inferred postal code: {inferred_zip_from_coords}")
                else:
                    st.sidebar.warning("Could not infer postal code from these coordinates. Please enter a zipcode manually if you know it.")
            else:
                st.sidebar.warning("Reverse-geocoding requires the 'geopy' package. Install with `pip install geopy` or enter zipcode manually.")
        except ValueError:
            st.sidebar.error("Latitude and longitude must be numeric.")
else:
    # Zipcode input (original flow)
    zipcode_input = st.sidebar.text_input("Enter Zipcode", value="55455")

# Common inputs (kept)
target_user = st.sidebar.number_input("Enter User ID (optional)", value=1, step=1)
fallback_radius = st.sidebar.slider("Nearby search radius (in zipcode steps)", min_value=1, max_value=10, value=3)

# Determine which zipcode to use: direct input or inferred from coordinates
used_zipcode_for_queries = None
if use_gps and inferred_zip_from_coords:
    used_zipcode_for_queries = inferred_zip_from_coords
elif use_gps and (lat_input.strip() != "" and lon_input.strip() != "" and not inferred_zip_from_coords):
    # user tried coords but inference failed
    st.warning("Coordinates provided but postal code couldn't be inferred. Please enter zipcode manually in the sidebar or try different coordinates.")
    used_zipcode_for_queries = None
elif not use_gps:
    used_zipcode_for_queries = zipcode_input
else:
    used_zipcode_for_queries = None  # user hasn't provided usable input yet

# UI: Global top movies
st.write("## 🔹 Global Top Movies")
st.dataframe(get_global_top_movies())

# Location top movies with fallback
st.write("## 🔹 Top Movies in your Location (with nearby fallback)")
if used_zipcode_for_queries is None:
    st.info("No valid zipcode available yet. Provide a zipcode in the sidebar, or provide coordinates and reverse-geocode them to get a zipcode.")
else:
    location_top, used_zip_for_top = get_location_top_movies(used_zipcode_for_queries, n=10, fallback_max_radius=fallback_radius)
    if location_top.empty:
        st.warning(f"No data found for `{used_zipcode_for_queries}` or nearby zipcodes within ±{fallback_radius}.")
    else:
        if used_zip_for_top != used_zipcode_for_queries:
            st.info(f"No data for `{used_zipcode_for_queries}` — showing top movies for nearby zipcode `{used_zip_for_top}` instead.")
        st.dataframe(location_top)

# Collaborative recommendations with fallback
st.write("## 🔹 Personalized Recommendations")
if used_zipcode_for_queries is None:
    st.info("No valid zipcode available for collaborative recommendations. Provide zipcode or coordinates in the sidebar.")
else:
    collab_recs, used_zip_for_collab = get_collab_recommendations(used_zipcode_for_queries, target_user, n=10, fallback_max_radius=fallback_radius)
    if used_zip_for_collab is None:
        st.warning(f"No location data available for `{used_zipcode_for_queries}` or nearby zipcodes within ±{fallback_radius}.")
    elif not collab_recs:
        if used_zip_for_collab != used_zipcode_for_queries:
            st.warning(f"Not enough data for collaborative recommendations in `{used_zipcode_for_queries}`. Checked nearby zipcode `{used_zip_for_collab}` but still not enough data for collaborative filtering.")
        else:
            st.warning("Not enough data for collaborative recommendations in this zipcode.")
    else:
        if used_zip_for_collab != used_zipcode_for_queries:
            st.info(f"No collaborative data for `{used_zipcode_for_queries}` — using users from nearby zipcode `{used_zip_for_collab}` for recommendations.")
        st.write(pd.DataFrame(collab_recs, columns=["Movie", "Score"]))

# Footer notes about geopy
st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- You can either enter a zipcode directly (original flow) or provide GPS coordinates (lat, lon).")
if not GEOPY_AVAILABLE:
    st.markdown("- Reverse-geocoding coordinates to postal codes requires `geopy`. Install it with: `pip install geopy`.")
st.markdown("- The app uses MovieLens 1M csv files and the zipcode field from the MovieLens users table. GPS -> zipcode is inferred via reverse-geocoding and then falls back to your existing zipcode-based logic (nearby zipcodes search).")
