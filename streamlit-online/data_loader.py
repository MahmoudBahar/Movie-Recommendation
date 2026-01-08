import io
import os
from typing import Optional, Tuple

import joblib
import pandas as pd
import requests
import streamlit as st

MOVIES_URL = "https://drive.usercontent.google.com/u/0/uc?id=1-SJ8oASjn4Ubbm5VlLGJXju_HfYe_HJp&export=download"
RATINGS_URL = "https://drive.usercontent.google.com/download?id=1I5OyLwZs26RZtbG0QUpi9H0ntZv9Ov7C&export=download&authuser=0&confirm=t&uuid=9da5ac1b-c3c3-410f-8d00-2f001585b64d&at=APvzH3ootqPTp4kiQ44Ew8EDit-R%3A1735991866003"
SVD_URL = "https://drive.usercontent.google.com/download?id=1-EZ3pUuoKn_C8RXuM-n8W8coo-20q4nB&export=download&authuser=0&confirm=t&uuid=f4f15388-b689-4fe7-aebe-b5ca56adda7f&at=APvzH3pCgFqBQ8ISNTL5zHrTIu5u%3A1735951544081"
WEIGHTED_SVD_URL = "https://drive.usercontent.google.com/download?id=1-9F142ZpJTyJCyP46NVwkButsVc7PfHC&export=download&authuser=0&confirm=t&uuid=7856b7f9-e65a-4c70-a719-8362fe70a431&at=APvzH3qxH32v9fKo2ASzcH4avHni%3A1735951432618"
KNN_URL = "https://drive.usercontent.google.com/u/0/uc?id=1-2j6yK1ajE37xifio-Eo1YLpbHsjLyLr&export=download"
TITLE_TO_IDX_URL = "https://drive.usercontent.google.com/u/0/uc?id=1-AoCFwt2MZ2ExVR1wt-LiT8IDehnvjUP&export=download"
IDX_TO_TITLE_URL = "https://drive.usercontent.google.com/u/0/uc?id=1-5kmSUkujuIKGAGHI76mRMZme6miXDCp&export=download"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _flag_from(value: Optional[object]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return None


def light_mode_enabled(default: bool = True) -> bool:
    """Return True when the app should avoid heavy data loads."""
    secret_value = None
    try:
        secret_value = st.secrets.get("light_mode", None)
    except Exception:
        secret_value = None
    env_value = os.getenv("MOVIE_APP_LIGHT_MODE")
    flag = _flag_from(secret_value)
    if flag is None:
        flag = _flag_from(env_value)
    if flag is None:
        return default
    return flag


def _download_bytes(url: str) -> io.BytesIO:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return io.BytesIO(response.content)


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    movies = pd.read_pickle(_download_bytes(MOVIES_URL))
    dtype_map = {
        "movieId": "int32",
        "popularity": "float32",
        "runtime": "float32",
        "weightedVoteAverage": "float32",
    }
    for col, dtype in dtype_map.items():
        if col in movies:
            movies[col] = pd.to_numeric(movies[col], errors="coerce").astype(dtype)
    if "release_date" in movies:
        movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    return movies.sort_values(
        by=["popularity", "weightedVoteAverage", "release_date"], ascending=False
    )


@st.cache_resource(show_spinner=False)
def load_content_bundle() -> Tuple[pd.DataFrame, pd.DataFrame, object, object, object]:
    movies = load_movies()
    features = movies.drop(
        [
            "original_language",
            "popularity",
            "runtime",
            "release_date",
            "poster_path",
            "weightedVoteAverage",
            "title",
            "movieId",
            "genres",
        ],
        axis="columns",
        errors="ignore",
    )
    knn_pl = joblib.load(_download_bytes(KNN_URL))
    title_to_idx = pd.read_pickle(_download_bytes(TITLE_TO_IDX_URL))
    idx_to_title = pd.read_pickle(_download_bytes(IDX_TO_TITLE_URL))
    return movies, features, knn_pl, title_to_idx, idx_to_title


@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame:
    ratings = pd.read_pickle(_download_bytes(RATINGS_URL))
    dtype_map = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    for col, dtype in dtype_map.items():
        if col in ratings:
            ratings[col] = pd.to_numeric(ratings[col], errors="coerce").astype(dtype)
    if "timestamp" in ratings:
        ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").astype(
            "int32"
        )
    return ratings


@st.cache_resource(show_spinner=False)
def load_cf_bundle() -> Tuple[pd.DataFrame, object, object]:
    ratings = load_ratings()
    algo = joblib.load(_download_bytes(SVD_URL))
    weighted_algo = joblib.load(_download_bytes(WEIGHTED_SVD_URL))
    return ratings, algo, weighted_algo
