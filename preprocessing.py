import pandas as pd
import re
import pickle
import nltk

from sklearn.utils import resample
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/dataset_labeled.csv")

# =====================
# BALANCING DATA
# =====================
df_netral = df[df.sentimen == "netral"]
df_positif = df[df.sentimen == "positif"]
df_negatif = df[df.sentimen == "negatif"]

df_negatif_up = resample(
    df_negatif,
    replace=True,
    n_samples=len(df_netral),
    random_state=42
)

df_balanced = pd.concat([df_netral, df_positif, df_negatif_up])
df = df_balanced.sample(frac=1, random_state=42)

print(df["sentimen"].value_counts())

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
        return "kosong"

    text = " ".join(tokens)
    text = stemmer.stem(text)
    return text

df["clean"] = df["komentar"].apply(preprocess)

# =====================
# TOKENIZER
# =====================
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean"])

sequences = tokenizer.texts_to_sequences(df["clean"])
X = pad_sequences(sequences, maxlen=100)

# =====================
# LABEL ENCODING
# =====================
label_map = {"negatif": 0, "netral": 1, "positif": 2}
y = df["sentimen"].map(label_map)

# =====================
# SAVE
# =====================
import os
os.makedirs("models", exist_ok=True)

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {"max_len": 100}
with open("models/config.pkl", "wb") as f:
    pickle.dump(config, f)

pd.DataFrame(X).to_csv("data/X.csv", index=False)
pd.DataFrame(y).to_csv("data/y.csv", index=False)

print("Preprocessing & balancing selesai!")