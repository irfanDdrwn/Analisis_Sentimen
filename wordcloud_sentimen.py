import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/dataset_labeled.csv")

# pastikan tidak ada NaN
df = df.dropna()

# =====================
# PISAHKAN SENTIMEN
# =====================
positif = " ".join(df[df["sentimen"] == "positif"]["komentar"].astype(str))
negatif = " ".join(df[df["sentimen"] == "negatif"]["komentar"].astype(str))
netral  = " ".join(df[df["sentimen"] == "netral"]["komentar"].astype(str))

# =====================
# FUNCTION WORDCLOUD
# =====================
def tampilkan_wordcloud(text, judul):
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)

    plt.figure()
    plt.imshow(wc)
    plt.title(judul)
    plt.axis("off")
    plt.show()

# =====================
# TAMPILKAN WORDCLOUD
# =====================
tampilkan_wordcloud(positif, "Wordcloud Sentimen Positif")
tampilkan_wordcloud(negatif, "Wordcloud Sentimen Negatif")
tampilkan_wordcloud(netral,  "Wordcloud Sentimen Netral")