
import streamlit as st
import pandas as pd
from PIL import Image
# Load banner image
@st.cache_resource
def load_image():
    image = Image.open("banner.jpg")
    image = image.crop((200, 600, 2120, 1200))
    return image

@st.cache_resource
def load_data():
    return pd.read_pickle('../MoviesProcessed.pkl').sort_values(by=['popularity', 'weightedVoteAverage', 'release_date'], ascending=False)

if 'df' not in st.session_state:
    st.session_state.df = load_data()

if "banner_image" not in st.session_state:
    st.session_state.banner = load_image()

# Display header with background image
st.image(st.session_state.banner, use_container_width=True)

st.title("🎥 Movie Magic: Your Personalized Recommendation Dashboard")
st.write(
    """
Welcome to **Movie Magic**, where discovering your next favorite movie is just a click away!
Dive into a world of cinema with personalized recommendations, insightful analytics, and an exploration of trends in the movie industry.
"""
)

# Add interactive buttons to direct users
st.subheader("🎬 Let the Magic Begin!")
col1, col2 = st.columns(2)

with col1:
    if st.button("💡 Get Recommendations"):
        st.switch_page("./Pages/model_page.py")

with col2:
    if st.button("📊 Explore Analytics"):
        st.switch_page("./Pages/analysis_page.py")

# Add a quick movie search box
st.markdown("---")
st.subheader("🎥 Quick Movie Search")
user_input = st.text_input("Search for a movie to explore its details:")
if user_input:
    output = st.session_state.df[st.session_state.df['title'].str.casefold().str.startswith(user_input.casefold())].head(30)
    output_temp = st.session_state.df[st.session_state.df['title'].str.contains(user_input, case=False)].head(30)
    output = pd.concat([output, output_temp]).drop_duplicates(subset=['title'], keep='first')
    # images = [st.image(image, width=200) for image in output['poster_path'].values]
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
            {image_elements}
        </div>
        """,
        unsafe_allow_html=True
    )

# Add a fun fact or inspirational quote about movies
st.markdown("---")
st.subheader("🎞️ Did You Know?")
st.markdown(
    """
    > *"Movies touch our hearts and awaken our vision, and change the way we see things. They take us to other places, they open doors and minds."*  
    — **Martin Scorsese**
    """
)
