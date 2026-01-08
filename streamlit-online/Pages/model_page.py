import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_cf_bundle, load_content_bundle
from mode_toggle import resolve_light_mode

# Streamlit app
st.title("🎥 Cool Movie Recommendation Page")
st.write("Choose a recommendation method below:")

light_mode = resolve_light_mode(key_prefix="model_")

# Tabs for different recommendation methods
tab1, tab2, tab3 = st.tabs(["👤 User-Based", "✨ Custom Preferences", "🎬 Movie Similarity"])

movies, features, knn_pl, title_to_idx, idx_to_title = load_content_bundle()


def scrollableElement(output, header):
    scroll_container_style = """
    <style>
    .scrollable-container {
        white-space: nowrap;
        overflow-x: auto;
    }

    .scrollable-container img {
        display: inline-block;
        margin-right: 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
    </style>
    """
    st.markdown(scroll_container_style, unsafe_allow_html=True)

    image_elements = ""
    if "rating" in output.columns:
        for img_path, title, rating in zip(
            output["poster_path"].values, output["title"].values, output["rating"].values
        ):
            image_elements += f'''<div style="display: inline-block; text-align: center;">
            <img src="{img_path}" alt="Some Image" width="120" height="160">
            <h6 style= "padding-top: 10px">{title}</h6>
            <h6>User rating: {rating}</h6>
            </div>'''
    else:
        for img_path, title in zip(output["poster_path"].values, output["title"].values):
            image_elements += f'''<div style="display: inline-block; text-align: center;">
            <img src="{img_path}" alt="Some Image" width="120" height="160">
            <h6 style= "padding-top: 10px">{title}</h6>
            </div>'''

    st.markdown(
        f"""
        <div class="scrollable-container">
        <h3>{header}</h3>
            {image_elements}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_cf_recommendations(user_id, ratings, algo, weighted_algo, top_n=10, weighted=False):
    user_ratings = ratings[ratings["userId"] == user_id]
    user_movies = user_ratings["movieId"].tolist()
    all_movies = movies["movieId"].tolist()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    predictions = [(weighted_algo if weighted else algo).predict(user_id, movie_id) for movie_id in movies_to_predict]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    top_predictions = predictions[:top_n]
    top_movie_ids = [pred.iid for pred in top_predictions]
    return movies[movies["movieId"].isin(top_movie_ids)]


def get_content_recommendation(movieName, numberOfMovies=10, direct=False):
    vector = None
    if direct:
        vector = movieName
    else:
        index = title_to_idx[movieName]
        vector = features.iloc[[index]].values
    vector = knn_pl.named_steps["L2 normalization"].transform(vector)
    _, indices = knn_pl.named_steps.KNN.kneighbors(vector.reshape(1, -1), numberOfMovies + 1)
    if direct:
        indices = indices[0]
    else:
        indices = [i for i in indices[0] if i != index]
    return movies[movies.title.isin(idx_to_title[indices].tolist())]


# 1. User-Based Recommendations
with tab1:
    st.subheader("👤 Recommendations for a User")
    if light_mode:
        st.info(
            "User-based recommendations are locked while light mode is on. "
            "Switch off the toggle above and enter the heavy-mode password (if set) to load the ratings + SVD models."
        )
    else:
        ratings, algo, weighted_algo = load_cf_bundle()
        user_choices = list(ratings["userId"].unique().astype(str))
        user_id = st.selectbox("Select a User ID", options=user_choices)
        useWeightedRating = st.toggle("Use Weighted Ratings", False)
        if st.button("Get Recommendations"):
            temp = ratings[ratings["userId"] == int(user_id)]
            watched_movies = movies[movies["movieId"].isin(temp["movieId"])]
            watched_movies = temp.merge(watched_movies, on="movieId").sort_values(
                ["rating", "popularity", "timestamp"], ascending=False
            )
            if not watched_movies.empty:
                scrollableElement(watched_movies, "Movies Watched by User")
            else:
                st.write("No movies watched by this user.")
            if not temp.empty:
                recommendations = get_cf_recommendations(
                    int(user_id), ratings, algo, weighted_algo, weighted=useWeightedRating
                )
                content_rec = [get_content_recommendation(title) for title in recommendations["title"]]
                if content_rec:
                    recommendations = pd.concat(
                        [recommendations, pd.concat(content_rec, ignore_index=True)], ignore_index=True
                    )
                scrollableElement(recommendations, "Recommended Movies")

# 2. Custom User Preferences
with tab2:
    st.subheader("✨ Recommendations Based on Your Preferences")
    genres = features.columns[:-1]
    selected_genres = st.multiselect("Choose your favorite genres:", genres)

    if st.button("Recommend for Me"):
        profile = []
        if selected_genres:
            for genre in features.columns:
                profile.append(1 if genre in selected_genres else 0)
            profile = np.array(profile, dtype=np.float32).reshape(1, -1)
            recommendations = get_content_recommendation(profile, 50, True).sort_values(
                ["weightedVoteAverage", "popularity"], ascending=False
            )
            scrollableElement(recommendations, "Recommended Movies")
        else:
            st.write("Please select at least one genre to get recommendations.")

# 3. Movie Similarity Recommendations
with tab3:
    st.subheader("🎬 Recommendations Based on a Movie")
    selected_movie = st.selectbox("Pick a Movie", options=movies["title"])
    st.image(movies[movies["title"] == selected_movie]["poster_path"].values[0], width=200)
    st.write(f"**Popularity:** {movies[movies['title'] == selected_movie]['popularity'].values[0]}")
    st.write(
        f"**Weighted Vote Average:** {movies[movies['title'] == selected_movie]['weightedVoteAverage'].values[0]:.2f}"
    )
    st.write(f"**Runtime:** {movies[movies['title'] == selected_movie]['runtime'].values[0]} minutes")
    st.write(f"**Release Date:** {movies[movies['title'] == selected_movie]['release_date'].values[0]}")

    if st.button("Find Similar Movies"):
        recommendations = get_content_recommendation(selected_movie, 50)
        scrollableElement(recommendations, "Recommended Movies")
