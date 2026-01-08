import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

from data_loader import get_data_source_label, load_movies, load_ratings
from mode_toggle import resolve_light_mode

light_mode = resolve_light_mode(key_prefix="analysis_")

movies = load_movies()
if movies is None:
    st.session_state["analysis_heavy_mode_unlocked"] = False
    st.session_state["analysis_light_mode_override"] = True
    st.stop()
ratings = None
if not light_mode:
    ratings = load_ratings()
    if ratings is None:
        # Loading ratings failed (likely memory). Force light mode for this session.
        st.session_state["analysis_heavy_mode_unlocked"] = False
        st.session_state["analysis_light_mode_override"] = True

# Page Title
st.title("📊 Analysis of Movies and Ratings")
st.caption(f"Data source: {get_data_source_label()}")

# Section 1: Quick Stats
st.header("🔍 Quick Stats")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Movies", len(movies))
with col2:
    st.metric("Total Ratings", "light mode" if ratings is None else len(ratings))
with col3:
    st.metric("Unique Users", "light mode" if ratings is None else ratings["userId"].nunique())

# Section 2: Top Genres Word Cloud
st.markdown("---")
st.header("🎭 Top Genres Word Cloud")
all_genres = movies["genres"].str.split("|").explode()
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_genres.dropna()))

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud.to_image(), interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

if ratings is None:
    st.markdown("---")
    st.info(
        "Ratings-heavy analytics are unavailable (light mode or not enough memory). "
        "Switch off the toggle above and enter the heavy-mode password (if set), or run locally with more RAM."
    )
else:
    # Section 3: Ratings Distribution
    st.markdown("---")
    st.header("⭐ Ratings Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ratings["rating"], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Section 4: Top 10 Most Rated Movies
    st.markdown("---")
    st.header("🏆 Top 10 Most Rated Movies")
    top_rated_movies = ratings.groupby("movieId").size().sort_values(ascending=False).head(10)
    top_rated_titles = movies.set_index("movieId").loc[top_rated_movies.index, "title"]
    top_rated_mean = ratings.groupby("movieId")["rating"].mean().loc[top_rated_movies.index]

    for idx, (title, count, avg) in enumerate(zip(top_rated_titles, top_rated_movies, top_rated_mean), 1):
        st.write(f"**{idx}. {title}** - {count} ratings, With average Rating: {avg:.2f}")

    # Section 5: Average Rating by Genre
    st.markdown("---")
    st.header("🎥 Average Rating by Genre")
    ratings_with_genres = pd.merge(ratings, movies[["movieId", "genres"]], on="movieId")
    ratings_with_genres["genres"] = ratings_with_genres["genres"].str.split("|")
    ratings_with_genres = ratings_with_genres.explode("genres")
    ratings_with_genres = ratings_with_genres.groupby("genres")["rating"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=ratings_with_genres.values, y=ratings_with_genres.index, ax=ax)
    ax.set_title("Average Rating by Genre")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

# Section 6: Interactive Genre Filter
st.markdown("---")
st.header("🔎 Explore Movies by Genre")
selected_genre = st.selectbox("Choose a Genre", all_genres.unique())

filtered_movies = movies[movies["genres"].str.contains(selected_genre, na=False)]
st.write(f"Movies in the genre **{selected_genre}**:")
st.dataframe(filtered_movies[["title", "genres"]])
