import streamlit as st
import pandas as pd
import joblib, requests, io
import numpy as np

# Streamlit app
st.title("🎥 Cool Movie Recommendation Page")
st.write("Choose a recommendation method below:")

# Tabs for different recommendation methods
tab1, tab2, tab3 = st.tabs(["👤 User-Based", "✨ Custom Preferences", "🎬 Movie Similarity"])

def scrollableElement(output, header):
    # CSS for horizontal scroll
    scroll_container_style = """
    <style>
    .scrollable-container {
        /* Force images to stay on one line and scroll horizontally */
        white-space: nowrap; 
        overflow-x: auto;  
    }

    /* Style each image (inline-block to stay side-by-side) */
    .scrollable-container img {
        display: inline-block;
        margin-right: 16px;  /* spacing between images */
        border: 1px solid #ccc; /* optional: add a border or some styling */
        border-radius: 4px;     /* optional: rounded corners */
    }
    </style>
    """

    # Inject CSS
    st.markdown(scroll_container_style, unsafe_allow_html=True)

    # --- 3) Build a single HTML string with all images ---
    image_elements = ""
    if 'rating' in output.columns:
        for img_path, title, rating in zip(output['poster_path'].values, output['title'].values, output['rating'].values):
            # Adjust width/height to fit your needs
            image_elements += f'''<div style="display: inline-block; text-align: center;">
            <img src="{img_path}" alt="Some Image" width="120" height="160">
            <h6 style= "padding-top: 10px">{title}</h6>
            <h6>User rating: {rating}</h6>
            </div>'''
    else:
        for img_path, title in zip(output['poster_path'].values, output['title'].values):
            # Adjust width/height to fit your needs
            image_elements += f'''<div style="display: inline-block; text-align: center;">
            <img src="{img_path}" alt="Some Image" width="120" height="160">
            <h6 style= "padding-top: 10px">{title}</h6>
            </div>'''

    # --- 4) Display them in one container ---
    st.markdown(
        f"""
        <div class="scrollable-container">
        <h3>{header}</h3>
            {image_elements}
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_data():
    return (pd.read_pickle(io.BytesIO(requests.get('https://drive.usercontent.google.com/u/0/uc?id=1-SJ8oASjn4Ubbm5VlLGJXju_HfYe_HJp&export=download').content)),
    pd.read_pickle(io.BytesIO(requests.get('https://drive.usercontent.google.com/download?id=1I5OyLwZs26RZtbG0QUpi9H0ntZv9Ov7C&export=download&authuser=0&confirm=t&uuid=9da5ac1b-c3c3-410f-8d00-2f001585b64d&at=APvzH3ootqPTp4kiQ44Ew8EDit-R%3A1735991866003').content)),
    joblib.load(io.BytesIO(requests.get('https://drive.usercontent.google.com/download?id=1-EZ3pUuoKn_C8RXuM-n8W8coo-20q4nB&export=download&authuser=0&confirm=t&uuid=f4f15388-b689-4fe7-aebe-b5ca56adda7f&at=APvzH3pCgFqBQ8ISNTL5zHrTIu5u%3A1735951544081').content)),
    joblib.load(io.BytesIO(requests.get('https://drive.usercontent.google.com/download?id=1-9F142ZpJTyJCyP46NVwkButsVc7PfHC&export=download&authuser=0&confirm=t&uuid=7856b7f9-e65a-4c70-a719-8362fe70a431&at=APvzH3qxH32v9fKo2ASzcH4avHni%3A1735951432618').content)),
    joblib.load(io.BytesIO(requests.get('https://drive.usercontent.google.com/u/0/uc?id=1-2j6yK1ajE37xifio-Eo1YLpbHsjLyLr&export=download').content)),
    pd.read_pickle(io.BytesIO(requests.get('https://drive.usercontent.google.com/u/0/uc?id=1-AoCFwt2MZ2ExVR1wt-LiT8IDehnvjUP&export=download').content)),
    pd.read_pickle(io.BytesIO(requests.get('https://drive.usercontent.google.com/u/0/uc?id=1-5kmSUkujuIKGAGHI76mRMZme6miXDCp&export=download').content)))

if any(i not in st.session_state for i in ["dfMovies", "dfRatings", "knn_pl", "algo", "titleToIdx", "idxToTitle", "features", "weightedAlgo"]):
    st.session_state.dfMovies, st.session_state.dfRatings, st.session_state.algo, st.session_state.weightedAlgo, st.session_state.knn_pl, st.session_state.titleToIdx, st.session_state.idxToTitle = load_data()
    st.session_state.features = st.session_state.dfMovies.drop(['original_language', 'popularity', 'runtime', 'release_date', 'poster_path', 'weightedVoteAverage', 'title', 'movieId', 'genres'], axis = 'columns')

def get_cf_recommendations(user_id, top_n=10, weighted=False):
    user_ratings = st.session_state.dfRatings[st.session_state.dfRatings['userId'] == user_id]
    user_movies = user_ratings['movieId'].tolist()
    all_movies = st.session_state.dfMovies['movieId'].tolist()
    movies_to_predict = list(set(all_movies) - set(user_movies))
    
    predictions = [(st.session_state.weightedAlgo if weighted else st.session_state.algo).predict(user_id, movie_id) for movie_id in movies_to_predict]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    top_predictions = predictions[:top_n]
    top_movie_ids = [pred.iid for pred in top_predictions]
    return st.session_state.dfMovies[st.session_state.dfMovies['movieId'].isin(top_movie_ids)]

def get_content_recommendation(movieName, numberOfMovies = 10, direct = False):
    vector = None
    if direct:
        vector = movieName
    else:
        index = st.session_state.titleToIdx[movieName]
        vector = st.session_state.features.iloc[[index]].values 
    vector = st.session_state.knn_pl.named_steps['L2 normalization'].transform(vector)
    _, indices = st.session_state.knn_pl.named_steps.KNN.kneighbors(vector.reshape(1, -1), numberOfMovies + 1)
    if direct:
        indices = indices[0]
    else:
        indices = list(filter(lambda i :True if i != index else False, indices[0]))
    return st.session_state.dfMovies[st.session_state.dfMovies.title.isin(st.session_state.idxToTitle[indices].tolist())]

# 1. User-Based Recommendations
with tab1:
    st.subheader("👤 Recommendations for a User")
    user_id = st.selectbox("Select a User ID", options=list(st.session_state.dfRatings["userId"].unique().astype(str)))
    useWeightedRating = st.toggle("Use Weighted Ratings", False)
    if st.button("Get Recommendations"):
        temp = st.session_state.dfRatings[st.session_state.dfRatings['userId'] == int(user_id)]
        watched_movies = st.session_state.dfMovies[st.session_state.dfMovies['movieId'].isin(temp['movieId'])]
        watched_movies = temp.merge(watched_movies, on='movieId').sort_values(['rating', 'popularity', 'timestamp'], ascending=False)
        if not watched_movies.empty:
            scrollableElement(watched_movies, "Movies Watched by User")
        else:
            st.write("No movies watched by this user.")
        if not temp.empty:
            recommendations = get_cf_recommendations(int(user_id), weighted = useWeightedRating)
            content_rec = []
            for title in recommendations['title']:
                content_rec.append(get_content_recommendation(title))
            recommendations = pd.concat([recommendations, pd.concat(content_rec, ignore_index=True)], ignore_index=True)
            scrollableElement(recommendations, "Recommended Movies")

# 2. Custom User Preferences
with tab2:
    st.subheader("✨ Recommendations Based on Your Preferences")
    genres = st.session_state.features.columns[:-1]
    selected_genres = st.multiselect("Choose your favorite genres:", genres)

    if st.button("Recommend for Me"):
        profile = []
        if selected_genres:
            for genre in st.session_state.features.columns:
                if genre in selected_genres:
                    profile.append(1)
                else:
                    profile.append(0)
            profile = np.array(profile).reshape(1, -1)
            recommendations = get_content_recommendation(profile, 50, True).sort_values(['weightedVoteAverage', 'popularity'], ascending=False)
            scrollableElement(recommendations, "Recommended Movies")
        else:
            st.write("Please select at least one genre to get recommendations.")

# 3. Movie Similarity Recommendations
with tab3:
    st.subheader("🎬 Recommendations Based on a Movie")
    selected_movie = st.selectbox("Pick a Movie", options=st.session_state.dfMovies["title"])
    st.image(st.session_state.dfMovies[st.session_state.dfMovies['title'] == selected_movie]['poster_path'].values[0], width=200)
    st.write(f"**Popularity:** {st.session_state.dfMovies[st.session_state.dfMovies['title'] == selected_movie]['popularity'].values[0]}")
    st.write(f"**Weighted Vote Average:** {st.session_state.dfMovies[st.session_state.dfMovies['title'] == selected_movie]['weightedVoteAverage'].values[0]:.2f}")
    st.write(f"**Runtime:** {st.session_state.dfMovies[st.session_state.dfMovies['title'] == selected_movie]['runtime'].values[0]} minutes")
    st.write(f"**Release Date:** {st.session_state.dfMovies[st.session_state.dfMovies['title'] == selected_movie]['release_date'].values[0]}")

    if st.button("Find Similar Movies"):
        recommendations = get_content_recommendation(selected_movie, 50)
        scrollableElement(recommendations, "Recommended Movies")
        # similar_movies = movies[movies["genres"].str.contains('|'.join(selected_genres.split('|')))]
        # similar_movies = similar_movies[similar_movies["title"] != selected_movie]
        # if not similar_movies.empty:
        #     st.write("**Movies Similar to Your Selection:**")
        #     st.table(similar_movies[["title", "genres"]])
        # else:
        #     st.write("No similar movies found.")