import streamlit as st
import numpy as np
import pickle
import re
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# download stopword sekali saja
nltk.download('stopwords')

# =====================
# LOAD MODEL & TOKENIZER
# =====================
model = load_model("models/model_lstm.h5")

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

    # kalau kosong
    if len(tokens) == 0:
        return "kosong"

    text = " ".join(tokens)
    text = stemmer.stem(text)
    return text

def predict_sentiment(text):
    original_text = text.lower()

    # =====================
    # RULE-BASED (PRIORITAS)
    # =====================
    positif_words = ["keren", "bagus", "mantap", "amazing", "good", "love"]
    negatif_words = ["jelek", "buruk", "benci", "bad", "boring", "hate"]

    if any(word in original_text for word in positif_words):
        return "Positif", 1.0

    if any(word in original_text for word in negatif_words):
        return "Negatif", 1.0

    # =====================
    # LSTM (SECONDARY)
    # =====================
    text = preprocess(text)
    seq = tokenizer.texts_to_sequences([text])

    if len(seq[0]) == 0:
        return "Netral", 0.5

    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded, verbose=0)[0]
    label = np.argmax(pred)

    labels = ["Negatif", "Netral", "Positif"]
    confidence = float(np.max(pred))

    return labels[label], confidence

# =====================
# UI STREAMLIT
# =====================
# =====================
# UI STREAMLIT
# =====================
st.title("Analisis Sentimen Komentar YouTube")
st.caption("Model LSTM - Justin Bieber Coachella")

user_input = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    if user_input.strip():
        hasil, _ = predict_sentiment(user_input)  # ambil label saja
        st.success(f"Hasil Sentimen: {hasil}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")