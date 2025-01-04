import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px

# Load data (example paths, update as necessary)
@st.cache_resource
def load_data():
    movies = pd.read_pickle("../MoviesProcessed.pkl")
    ratings = pd.read_pickle("../RatingsProcessed.pkl")
    return movies, ratings

if any([var not in st.session_state for var in ["dfMovies", "dfRatings"]]):
    st.session_state.dfMovies, st.session_state.dfRatings = load_data()

# Page Title
st.title("📊 Analysis of Movies and Ratings")

# Section 1: Quick Stats
st.header("🔍 Quick Stats")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Movies", len(st.session_state.dfMovies))
with col2:
    st.metric("Total Ratings", len(st.session_state.dfRatings))
with col3:
    st.metric("Unique Users", st.session_state.dfRatings['userId'].nunique())

# Section 2: Top Genres Word Cloud
st.markdown("---")
st.header("🎭 Top Genres Word Cloud")
@st.cache_resource
def get_all_genres():
    return st.session_state.dfMovies['genres'].str.split('|').explode()
@st.cache_resource
def workcloud_genres():
    all_genres = get_all_genres()
    return WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_genres.dropna()))
wordcloud = workcloud_genres()

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Section 3: Ratings Distribution
st.markdown("---")
st.header("⭐ Ratings Distribution")
@st.cache_resource
def ratings_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(st.session_state.dfRatings['rating'], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    return fig
st.pyplot(ratings_distribution())

# Section 4: Top 10 Most Rated Movies
st.markdown("---")
st.header("🏆 Top 10 Most Rated Movies")
top_rated_movies = st.session_state.dfRatings.groupby('movieId').size().sort_values(ascending=False).head(10)
top_rated_titles = st.session_state.dfMovies.set_index('movieId').loc[top_rated_movies.index, 'title']
top_rated_mean = st.session_state.dfRatings.groupby('movieId')['rating'].mean().loc[top_rated_movies.index]

# Display in columns
for idx, (title, count, avg) in enumerate(zip(top_rated_titles, top_rated_movies, top_rated_mean), 1):
    st.write(f"**{idx}. {title}** - {count} ratings, With average Rating: {avg:.2f}")

# Section 5: Average Rating by Genre
st.markdown("---")
st.header("🎥 Average Rating by Genre")
@st.cache_resource
def avg_rating_by_genre():
    ratings_with_genres = pd.merge(st.session_state.dfRatings, st.session_state.dfMovies[['movieId', 'genres']], on='movieId')
    ratings_with_genres['genres'] = ratings_with_genres['genres'].str.split('|')
    ratings_with_genres = ratings_with_genres.explode('genres')
    ratings_with_genres = ratings_with_genres.groupby('genres')['rating'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=ratings_with_genres.values, y=ratings_with_genres.index, ax=ax)
    ax.set_title("Average Rating by Genre")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    return fig

st.pyplot(avg_rating_by_genre())

# Section 6: Interactive Genre Filter
st.markdown("---")
st.header("🔎 Explore Movies by Genre")
selected_genre = st.selectbox("Choose a Genre", get_all_genres().unique())

filtered_movies = st.session_state.dfMovies[st.session_state.dfMovies['genres'].str.contains(selected_genre, na=False)]
st.write(f"Movies in the genre **{selected_genre}**:")
st.dataframe(filtered_movies[['title', 'genres']])
