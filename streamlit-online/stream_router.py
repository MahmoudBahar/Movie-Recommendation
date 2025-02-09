import streamlit as st
import pandas as pd

st.set_page_config(page_icon= "🎬", page_title="Movie Recommendation", layout="wide", initial_sidebar_state='collapsed')

pg = st.navigation([st.Page("./Pages/overview_page.py",
                            title="Overview",
                            icon=":material/overview:",
                            url_path = 'Overview',
                            default=True
                           ),
                    st.Page("./Pages/model_page.py",
                            title="Movie Recommender",
                            icon=":material/map:",
                            url_path = 'Recommendations',
                            default=False
                           ),
                    st.Page("./Pages/analysis_page.py",
                            title="Analysis",
                            url_path = 'Analytics',
                            icon=":material/analytics:",
                            default=False
                           )])
pg.run()
