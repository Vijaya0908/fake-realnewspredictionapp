import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('random_forest_classifier.joblib')  # Replace with your actual model file
vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Replace with your actual vectorizer file

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text):
    tokens = word_tokenize(text)
    return " ".join(tokens)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(filtered_words)

def predict_fake_real_news(text):
    # Preprocess the input text
    text = preprocess_text(text)
    text = tokenize_text(text)
    text = remove_stopwords(text)

    # Vectorize the text
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)

    return prediction[0]

# Streamlit UI
st.set_page_config(
    page_title="Fake and Real News Prediction App",
    page_icon=":newspaper:",
    layout="wide"
)

# Streamlit UI
st.title("Fake and Real News Prediction App")

# Input text box for user to enter news text
user_input = st.text_area("Enter the news text:")

# Make prediction when the user clicks the "Predict" button
if st.button("Predict"):
    if user_input:
        prediction = predict_fake_real_news(user_input)

        # Display the prediction result
        if prediction == 0:
            st.success("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")
    else:
        st.warning("Please enter some text for prediction.")

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info(
    "This web app predicts whether a given news article is fake or real. "
    "It uses a machine learning model trained on a dataset of labeled news articles."
)

