import pandas as pd
from textblob import TextBlob

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/dataset_youtube_justin_coachella.csv")

print("Kolom dataset:", df.columns)

# pastikan kolom komentar benar
kolom_teks = "komentar" if "komentar" in df.columns else df.columns[0]

# =====================
# LABELING
# =====================
def label_sentiment(text):
    if pd.isna(text):
        return "netral"
    try:
        score = TextBlob(str(text)).sentiment.polarity
        if score > 0:
            return "positif"
        elif score < 0:
            return "negatif"
        else:
            return "netral"
    except:
        return "netral"

df["sentimen"] = df[kolom_teks].apply(label_sentiment)

# =====================
# SAVE
# =====================
df.to_csv("data/dataset_labeled.csv", index=False)

print("Labeling selesai!")
print(df.head())
