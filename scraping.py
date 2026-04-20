from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

# inisialisasi downloader
downloader = YoutubeCommentDownloader()

# URL video YouTube
url = "https://youtu.be/He9WmjUvpJ8"

# ambil komentar
comments = downloader.get_comments_from_url(url)

data = []

for i, comment in enumerate(comments):
    if i >= 10000:  # target sesuai modul
        break
    
    data.append({
        "platform": "youtube",
        "komentar": comment["text"],
        "timestamp": comment["time"]
    })

# buat dataframe
df = pd.DataFrame(data)

# simpan
df.to_csv("dataset_youtube_justin_coachella.csv", index=False)

print("Jumlah data:", len(df))
df.head()
df["sentimen"] = ""
from textblob import TextBlob

def label_sentiment(text):
    try:
        score = TextBlob(text).sentiment.polarity
        if score > 0:
            return "positif"
        elif score < 0:
            return "negatif"
        else:
            return "netral"
    except:
        return "netral"

df["sentimen"] = df["komentar"].apply(label_sentiment)

df.to_csv("dataset_labeled.csv", index=False)