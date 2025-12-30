# Location-Based-Movie-Recommendation-System

A Python-based movie recommendation system that suggests movies based on user location and collaborative filtering techniques.

This project uses the MovieLens 1M dataset and applies recommender system algorithms to recommend movies that are popular or relevant to users in a given geographic region.

🧠 Features

📍 Location-Aware Recommendations — Suggest movies that are trending or relevant based on the user’s location (e.g., city or region).

🎥 Collaborative Filtering — Uses user rating patterns for recommendation.

📊 Interactive UI — Simple interface (Streamlit/Flask) for user input and movie suggestions.

📦 Python Modules — Built using popular Python libraries such as Pandas, Scikit-Learn, etc.

📂 Project Structure
Location-Based-Movie-Recommendation-System/
├── ml-1m/                         # MovieLens 1M dataset
├── app2.py                       # Main application script
├── README.md                     # (You are here!)
├── requirements.txt              # Python dependencies
└── .gitattributes


The dataset folder ml-1m contains the MovieLens 1M ratings and user location metadata.

🛠️ Installation

Clone the Repository

git clone https://github.com/Abhishek464/Location-Based-Movie-Recommendation-System.git
cd Location-Based-Movie-Recommendation-System


Create & Activate Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt

🚀 How to Run

You can run the app locally with:

python app2.py


Once the app is running, open your browser and navigate to:

http://localhost:5000


(or the appropriate port if using Streamlit/Flask)

📊 Example Usage

Enter your location/city

Choose any additional filters (genre, rating, etc.)

Click Get Recommendations

A list of personalized movie suggestions will appear

📦 Dependencies

The project uses:

Python 3.8+

Pandas — for data processing

Scikit-Learn — for recommender algorithms

Flask / Streamlit — for the web interface

Install all dependencies with:

pip install -r requirements.txt

🧪 Dataset

The system uses the MovieLens 1M dataset, which contains approximately 1 million ratings from 6000 users for 4000+ movies. This dataset includes user demographic data, which can be used for location-based filtering.

📈 Recommendation Approach

The system can combine:

Collaborative Filtering — suggests movies based on similar users’ ratings

Location Influence — optionally weights recommendations by local popularity or regional trends

Hybrid Techniques (future extension)

(You can describe the exact algorithm here once you implement it.)
