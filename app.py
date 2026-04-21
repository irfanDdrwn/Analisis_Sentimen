import streamlit as st
import numpy as np
import pickle
import re
import nltk
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================
# DOWNLOAD STOPWORDS
# =====================
nltk.download('stopwords')

# =====================
# LOAD MODEL & TOKENIZER
# =====================
model = None

try:
    if os.path.exists("models/model_lstm.h5"):
        model = load_model("models/model_lstm.h5")
except:
    model = None

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/config.pkl", "rb") as f:
    config = pickle.load(f)

MAX_LEN = config["max_len"]

# =====================
# PREPROCESSING
# =====================
stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    tokens = [w for w in text.split() if w not in stop_words]

    if len(tokens) == 0:
        return ""

    text = " ".join(tokens)
    return stemmer.stem(text)

# =====================
# PREDICTION FUNCTION
# =====================
def predict_sentiment(text):

    original_text = text.lower()

    # =====================
    # RULE BASED (PRIORITY)
    # =====================
    positif_words = ["keren", "bagus", "mantap", "amazing", "good", "love"]
    negatif_words = ["jelek", "buruk", "benci", "bad", "boring", "hate"]

    if any(w in original_text for w in positif_words):
        return "Positif"

    if any(w in original_text for w in negatif_words):
        return "Negatif"

    # =====================
    # LSTM MODEL
    # =====================
    if model is None:
        return "Netral"

    text_clean = preprocess(text)

    if text_clean == "":
        return "Netral"

    seq = tokenizer.texts_to_sequences([text_clean])

    if len(seq[0]) == 0:
        return "Netral"

    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)[0]

    labels = ["Negatif", "Netral", "Positif"]
    return labels[np.argmax(pred)]

# =====================
# STREAMLIT UI
# =====================
st.title("Analisis Sentimen Komentar YouTube")
st.caption("Model LSTM - Justin Bieber Coachella")

user_input = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    if user_input.strip():
        hasil = predict_sentiment(user_input)
        st.success(f"Hasil Sentimen: {hasil}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")