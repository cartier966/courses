"""
Lessons 6 & 7 Interactive Student File
Topic: Streamlit + Stateful Adaptive Recommendation App

How to use:
1. Open a terminal in the project folder.
2. Run this file with Streamlit:
   .venv\\Scripts\\python.exe -m streamlit run streamlit_adaptive_recommender_student_app.py
3. Work through each section in order.
"""

import streamlit as st

# ============================================================
# SECTION 1: BASIC STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Streamlit Learning App",
    layout="wide"
)

st.title("Lessons 6 & 7: Streamlit Adaptive Recommender")
st.write("This file teaches Streamlit step by step.")


# ============================================================
# SECTION 2: DISPLAYING INFORMATION
# ============================================================

st.header("Section 1: Displaying Information")

st.subheader("Movie Recommendation Dashboard")
st.write("Streamlit lets us display Python output as a web app.")
st.metric("Example Confidence Score", "92%")


# ============================================================
# SECTION 3: USER INPUTS
# ============================================================

st.header("Section 2: User Inputs")

name = st.text_input("What is your name?")

favorite_genre = st.selectbox(
    "Choose your favorite genre",
    ["Action", "Comedy", "Sci-Fi", "Drama", "Animation"]
)

watch_time = st.slider(
    "How many hours do you watch movies per week?",
    0,
    40,
    5
)

if name:
    st.write(f"Hi {name}!")
    st.write(f"Your favorite genre is {favorite_genre}.")
    st.write(f"You watch around {watch_time} hours per week.")


# ============================================================
# SECTION 4: SIMPLE RECOMMENDATION LOGIC
# ============================================================

st.header("Section 3: Simple Recommendation Logic")

genre_match = st.slider("Genre Match Score", 0, 10, 5)
popularity = st.slider("Popularity Score", 0, 10, 5)

score = genre_match * 0.7 + popularity * 0.3

st.write("Recommendation Score:", round(score, 2))

if score >= 7:
    st.success("Strong Recommendation")
elif score >= 4:
    st.info("Maybe Recommend")
else:
    st.warning("Do Not Recommend")


# ============================================================
# SECTION 5: SESSION STATE
# ============================================================

st.header("Section 4: Session State")

if "demo_likes" not in st.session_state:
    st.session_state.demo_likes = 0

def add_demo_like():
    st.session_state.demo_likes += 1

st.button("Like Demo", on_click=add_demo_like)

st.write("Demo Likes:", st.session_state.demo_likes)


# ============================================================
# SECTION 6: ADAPTIVE MOVIE RECOMMENDER PROJECT
# ============================================================

st.divider()
st.header("Final Project: Adaptive Movie Recommender")

movies = [
    {"id": 1, "title": "Interstellar", "genre": "Sci-Fi", "base_score": 8.8, "emoji": "🚀"},
    {"id": 2, "title": "The Dark Knight", "genre": "Action", "base_score": 9.0, "emoji": "🦇"},
    {"id": 3, "title": "Finding Nemo", "genre": "Animation", "base_score": 7.7, "emoji": "🐠"},
    {"id": 4, "title": "Superbad", "genre": "Comedy", "base_score": 7.6, "emoji": "😂"},
    {"id": 5, "title": "The Godfather", "genre": "Drama", "base_score": 9.2, "emoji": "🎩"},
    {"id": 6, "title": "Dune", "genre": "Sci-Fi", "base_score": 8.4, "emoji": "🏜️"},
    {"id": 7, "title": "John Wick", "genre": "Action", "base_score": 8.1, "emoji": "💥"},
    {"id": 8, "title": "Toy Story", "genre": "Animation", "base_score": 8.3, "emoji": "🧸"}
]

# TODO:
# Add at least 3 more movies to the list above.


# Initialize session state

if "genre_weights" not in st.session_state:
    st.session_state.genre_weights = {
        "Action": 1.0,
        "Comedy": 1.0,
        "Sci-Fi": 1.0,
        "Drama": 1.0,
        "Animation": 1.0
    }

if "liked_movies" not in st.session_state:
    st.session_state.liked_movies = []

if "disliked_movies" not in st.session_state:
    st.session_state.disliked_movies = []


# Callback functions

def like_movie(movie):
    genre = movie["genre"]
    st.session_state.genre_weights[genre] += 0.2

    if movie["title"] not in st.session_state.liked_movies:
        st.session_state.liked_movies.append(movie["title"])


def dislike_movie(movie):
    genre = movie["genre"]

    # Advanced improvement:
    # Use max(0, value) to prevent the weight from going negative.
    st.session_state.genre_weights[genre] -= 0.2

    if movie["title"] not in st.session_state.disliked_movies:
        st.session_state.disliked_movies.append(movie["title"])


def reset_preferences():
    st.session_state.genre_weights = {
        "Action": 1.0,
        "Comedy": 1.0,
        "Sci-Fi": 1.0,
        "Drama": 1.0,
        "Animation": 1.0
    }
    st.session_state.liked_movies = []
    st.session_state.disliked_movies = []


# Scoring function

def calculate_final_score(movie):
    genre = movie["genre"]
    weight = st.session_state.genre_weights[genre]
    final_score = movie["base_score"] * weight
    return final_score


# Calculate and rank movies

for movie in movies:
    movie["final_score"] = calculate_final_score(movie)

ranked_movies = sorted(
    movies,
    key=lambda movie: movie["final_score"],
    reverse=True
)


# Sidebar

st.sidebar.header("User Preference Weights")
st.sidebar.write(st.session_state.genre_weights)

st.sidebar.header("Liked Movies")
st.sidebar.write(st.session_state.liked_movies)

st.sidebar.header("Disliked Movies")
st.sidebar.write(st.session_state.disliked_movies)

st.sidebar.button("Reset Preferences", on_click=reset_preferences)

st.sidebar.subheader("Preference Chart")
st.sidebar.bar_chart(st.session_state.genre_weights)


# Display recommendation cards

st.subheader("Recommended Movies")

cols = st.columns(3)

for index, movie in enumerate(ranked_movies):
    col = cols[index % 3]

    with col:
        with st.container(border=True):
            st.subheader(f"{movie['emoji']} {movie['title']}")
            st.write(f"Genre: {movie['genre']}")
            st.metric("Base Score", movie["base_score"])
            st.metric("Personalized Score", round(movie["final_score"], 2))

            like_col, dislike_col = st.columns(2)

            with like_col:
                st.button(
                    "Like",
                    key=f"like_{movie['id']}",
                    on_click=like_movie,
                    args=(movie,)
                )

            with dislike_col:
                st.button(
                    "Dislike",
                    key=f"dislike_{movie['id']}",
                    on_click=dislike_movie,
                    args=(movie,)
                )


# Explanation

st.divider()

st.subheader("How This App Learns")
st.write(
    "When you like a movie, the app increases the weight for that movie's genre. "
    "When you dislike a movie, the app decreases the weight for that genre. "
    "The app then recalculates each movie's personalized score and reranks the recommendations."
)

st.subheader("Student Tasks")
st.write(
    """
    1. Add at least 3 more movies.
    2. Add one new genre.
    3. Make sure the new genre has a starting weight.
    4. Prevent genre weights from going below 0.
    5. Stress test the app by liking one genre 5 times, then another genre 5 times.
    """
)
