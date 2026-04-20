import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# =====================
# LOAD DATA
# =====================
X = pd.read_csv("data/X.csv").values
y = pd.read_csv("data/y.csv").values

# ubah ke categorical
y = to_categorical(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# MODEL LSTM (DITINGKATKAN)
# =====================
model = Sequential()

model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# =====================
# COMPILE
# =====================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =====================
# TRAINING
# =====================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =====================
# EVALUASI
# =====================
loss, acc = model.evaluate(X_test, y_test)
print("Akurasi:", acc)

# =====================
# SAVE MODEL
# =====================
model.save("models/model_lstm.h5")

print("Model berhasil disimpan!")