import pandas as pd
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("data/dataset_labeled.csv")

# =====================
# HITUNG DISTRIBUSI SENTIMEN
# =====================
sentimen_counts = df["sentimen"].value_counts()

print(sentimen_counts)

# =====================
# PIE CHART
# =====================
plt.figure()
plt.pie(
    sentimen_counts,
    labels=sentimen_counts.index,
    autopct='%1.1f%%'
)
plt.title("Distribusi Sentimen Komentar YouTube")
plt.show()

# =====================
# BAR CHART
# =====================
plt.figure()
plt.bar(sentimen_counts.index, sentimen_counts.values)
plt.title("Jumlah Sentimen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah")
plt.show()
plt.title("Distribusi Sentimen Komentar YouTube\nKonser Justin Bieber di Coachella")
plt.savefig("hasil_pie_chart.png")
plt.savefig("hasil_bar_chart.png")
