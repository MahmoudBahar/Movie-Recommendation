# Movie Recommendation System

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMahmoudBahar%2FMovie-Recommendation&count_bg=%2379C83D&title_bg=%23555555&icon=themoviedatabase.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Overview

This **Movie Recommendation System** employs two key techniques: **Collaborative Filtering** using **Singular Value Decomposition (SVD)** to analyze user preferences, and **Content-Based Filtering** using **K-Nearest Neighbors (KNN)** to identify similar movies based on metadata. The system is deployed via **Streamlit**, offering an intuitive and interactive web interface for users to search, explore, and receive recommendations.

This project integrates **Time Decay Ratings** into user-based recommendations, ensuring that recent interactions are weighted more significantly for improved accuracy. Additionally, the **Hybrid Approach** in Streamlit seamlessly blends **Content-Based Filtering** and **Collaborative Filtering**, leveraging the strengths of both methods to provide more personalized and relevant movie recommendations.

## Features
- **Search Functionality**: Users can search for movies in the **Overview** page.
- **Recommendation System**:
  - **User-based recommendations (Collaborative Filtering - SVD with Time Decay Ratings)**
  - **Content-based recommendations (KNN)**
  - **Hybrid Recommendations**: Combining **content-based and collaborative filtering** for better user predictions.
- **Analysis Page**: Provides insights into the recommendation system and its underlying data.
- **Interactive UI**: Built with **Streamlit** for easy navigation and visualization.
- **Search Functionality**: Users can search for movies in the **Overview** page.
- **Recommendation System**:
  - **User-based recommendations (Collaborative Filtering - SVD)**
  - **Content-based recommendations (KNN)**
- **Analysis Page**: Provides insights into the recommendation system and its underlying data.
- **Interactive UI**: Built with **Streamlit** for easy navigation and visualization.

## How It Works
1. **Collaborative Filtering (SVD)**
   - Based on user ratings and interactions.
   - Finds latent factors to predict user preferences.
2. **Content-Based Filtering (KNN)**
   - Uses movie metadata (genres, descriptions, etc.) to find similar movies.
   - Computes similarity between movies for better recommendations.
3. **Streamlit Web App**
   - Users can **search**, **get recommendations**, and **analyze** trends in an intuitive interface.

## Installation & Setup
### Prerequisites
Ensure you have the following installed. If not, you can install them using:

```bash
pip install streamlit pandas numpy scikit-learn scikit-surprise
```

- Python 3.x
- Streamlit
- Pandas, NumPy, Scikit-learn, scikit-surprise

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the project directory and run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the **Overview Page** to search for a movie.
2. Navigate to the **Recommendation Page** to get movie recommendations:
   - Choose between **content-based** or **user-based** recommendations.
3. Visit the **Analysis Page** for data insights and performance metrics.

## Demo
This demo video showcases the functionality of the Streamlit-based movie recommendation system, including searching for movies, receiving personalized recommendations, and exploring data insights through the analysis page.

[Watch the video demo here](https://github.com/user-attachments/assets/0ac760ef-8d1e-4994-b999-e72f9200aa95)

## License
This project is licensed under the MIT License.

## Contact
For any queries or suggestions, feel free to reach out:
- **Email**: mahmoudbahar585@outlook.com
- **GitHub**: [MahmoudBahar](https://github.com/MahmoudBahar)

