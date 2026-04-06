import streamlit as st
import pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load model, scaler, feature columns, director encoding ---
with open('movie_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('director_encoding.pkl', 'rb') as f:
    director_encoding = pickle.load(f)

description_df = pd.read_csv('description_data.csv')
global_mean = 0.335

# --- Page config ---
st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

# --- Header ---
st.markdown("""
    <h1 style='text-align: center; color: #E50914;'>🎬 Movie Rating Predictor</h1>
    <p style='text-align: center; color: grey;'>Will your movie be a hit or a miss?</p>
    <hr>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🎬 Predict", "☁️ Word Cloud", "ℹ️ Model Info"])

with tab1:
    # --- Input form ---
    st.subheader("Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2020)
        duration = st.number_input("Duration (minutes)", min_value=30, max_value=300, value=120)
        votes = st.number_input("Expected Votes", min_value=0, max_value=3000000, value=50000)
        director = st.text_input("Director Name", placeholder="e.g. Christopher Nolan")

    with col2:
        genre = st.selectbox("Primary Genre", [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
            'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western'
        ])
        certificate = st.selectbox("Certificate", [
            'G', 'PG', 'PG-13', 'R', 'U', 'U/A', 'UA', '12', '15+', '18', 'Unknown'
        ])

    # --- Predict button ---
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🎬 Predict Rating", use_container_width=True)

    if predict_btn:
        # Build input dataframe
        input_data = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

        # Fill numerical values
        input_data['Year'] = year
        input_data['Duration (min)'] = duration
        input_data['Votes'] = votes

        # Fill genre
        genre_col = f'Genre_{genre}'
        if genre_col in input_data.columns:
            input_data[genre_col] = 1

        # Fill certificate
        cert_col = f'Certificate_{certificate}'
        if cert_col in input_data.columns:
            input_data[cert_col] = 1

        # Director encoding
        if director.strip() != "":
            director_value = director_encoding.get(director.strip(), global_mean)
        else:
            director_value = global_mean
        input_data['Director_encoded'] = director_value

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # --- Show result ---
        st.markdown("<hr>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""
                <div style='background-color:#1db954; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:white;'>🟢 Good Movie!</h2>
                    <p style='color:white;'>Predicted to score 7.0 or above on IMDb</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color:#E50914; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:white;'>🔴 Bad Movie!</h2>
                    <p style='color:white;'>Predicted to score below 7.0 on IMDb</p>
                </div>
            """, unsafe_allow_html=True)

        # --- Confidence gauge ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prediction Confidence")
        col1, col2 = st.columns(2)
        col1.metric("🔴 Bad Movie", f"{probability[0]*100:.1f}%")
        col2.metric("🟢 Good Movie", f"{probability[1]*100:.1f}%")
        st.progress(float(probability[1]))

        # --- Similar movies ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎬 Similar Movies from Dataset")
        similar = description_df[description_df['Target'] == prediction].sample(5, random_state=42)
        for i, row in similar.iterrows():
            label = "🟢 Good" if row['Target'] == 1 else "🔴 Bad"
            st.markdown(f"**{label} — {row['Title']}**")
            st.markdown(f"{row['Description'][:150]}...")
            st.markdown("---")

with tab2:
    st.subheader("☁️ Word Cloud of Movie Descriptions")
    option = st.radio("Show words from:", ["Good Movies", "Bad Movies", "All Movies"])

    if option == "Good Movies":
        text = " ".join(description_df[description_df['Target'] == 1]['Description'].dropna())
    elif option == "Bad Movies":
        text = " ".join(description_df[description_df['Target'] == 0]['Description'].dropna())
    else:
        text = " ".join(description_df['Description'].dropna())

    wordcloud = WordCloud(width=800, height=400,
                         background_color='black',
                         colormap='Reds').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tab3:
    st.subheader("ℹ️ Model Information")

    st.markdown("### 🤖 Model Architecture")
    st.markdown("""
    - **Algorithm:** Random Forest Classifier
    - **Library:** scikit-learn
    - **Task:** Binary Classification (Good vs Bad movie)
    - **Threshold:** IMDb rating ≥ 7.0 = Good, < 7.0 = Bad
    """)

    st.markdown("### ⚙️ Key Hyperparameters")
    params = {
        "Parameter": ["n_estimators", "class_weight", "random_state"],
        "Value": ["50", "balanced", "42"],
        "Reason": [
            "Reduced from 100 to fit GitHub 25MB limit",
            "Handles class imbalance (66% Bad, 34% Good)",
            "Ensures reproducible results"
        ]
    }
    st.table(pd.DataFrame(params))

    st.markdown("### 🧪 Feature Engineering")
    features = {
        "Feature": ["Year", "Duration (min)", "Votes", "Genre_*", "Certificate_*", "Director_encoded"],
        "Type": ["Numerical", "Numerical", "Numerical", "One-hot Encoded", "One-hot Encoded", "Target Encoded"],
        "Description": [
            "Release year of the movie",
            "Runtime in minutes",
            "Number of IMDb votes",
            "22 genre categories",
            "27 certificate categories",
            "Director reputation score based on past movie ratings"
        ]
    }
    st.table(pd.DataFrame(features))
    st.caption("Total features: 53 (52 from one-hot encoding + 1 Director encoded)")

    st.markdown("### 📊 Training Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "9,596")
    col2.metric("Training Set", "7,676 (80%)")
    col3.metric("Test Set", "1,920 (20%)")
    st.markdown("""
    - **Source:** IMDB Movies Dataset (Kaggle)
    - **Original size:** 10,000 movies, 15 features
    - **After cleaning:** 9,596 movies, 53 features
    """)

    st.markdown("### 🏆 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "75.7%")
    col2.metric("Macro F1 Score", "0.71")
    col3.metric("CV Mean", "88.3%")

    st.markdown("### 🔮 Prediction Pipeline")
    st.markdown("""
    1. User inputs movie details
    2. Categorical features are one-hot encoded
    3. Director name is target encoded using training data
    4. All features scaled using StandardScaler
    5. Random Forest predicts Good or Bad
    6. Confidence scores displayed using predict_proba
    """)

    st.markdown("### 📁 Dataset")
    st.markdown("""
    - **Source:** Equinor Volve Open Dataset (Kaggle)
    - **Author:** Aman Barthwal
    - **URL:** https://www.kaggle.com/datasets/amanbarthwal/imdb-movies-dataset
    """)