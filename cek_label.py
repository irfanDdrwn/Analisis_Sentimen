import pandas as pd

# load dataset
df = pd.read_csv("data/dataset_labeled.csv")

# tampilkan jumlah tiap label
print("Distribusi Sentimen:")
print(df["sentimen"].value_counts())

# tampilkan persen
print("\nPersentase:")
print(df["sentimen"].value_counts(normalize=True) * 100)