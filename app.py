import streamlit as st
import joblib
import os
import requests
import numpy as np

# Function to download large model from Hugging Face
@st.cache_data
def download_model():
    url = "https://huggingface.co/naziiiii/Sentiments/blob/main/voting_model.pkl"
    filename = "voting_model.pkl"
    if not os.path.exists(filename):
        with st.spinner("Downloading model..."):
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
    return filename

# Download and load model
model_path = download_model()
model = joblib.load(model_path)
