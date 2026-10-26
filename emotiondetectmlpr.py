import os
import re
import joblib

import pandas as pd
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text_medium(s: str) -> str:
    s = str(s).lower()
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    s = re.sub(r"[^a-z0-9\s!?']", " ", s)  # keep ! ? '
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


st.title("Emotion Detector")

model_choice = st.selectbox(
    "Choose a model",
    ["Naive Bayes", "GRU", "LSTM"]
)


DATA_PATH = "archive/go_emotions_dataset.csv"
df = pd.read_csv(DATA_PATH)

df["text"] = df["text"].fillna("").astype(str)

X_text = df["text"].astype(str)
X_text_clean = X_text.apply(clean_text_medium)

emotion_cols = [
    col for col in df.columns
    if col not in ["id", "text", "example_very_unclear"]
]

y = df[emotion_cols].values.astype(np.float32)
num_labels = y.shape[1]


label_counts = df[emotion_cols].sum()
label_counts_safe = label_counts.replace(0, 1)

pos_weight = (len(df) - label_counts_safe) / label_counts_safe
pos_weight_tensor = torch.tensor(pos_weight.values, dtype=torch.float32)
pos_weight_tensor = torch.clamp(pos_weight_tensor, min=1.0, max=8.0)


NB_MODEL_PATH = "nb_model.pkl"
GRU_PATH = "gru_model.pt"
LSTM_PATH = "lstm_model.pt"


def train_or_load_naive_bayes():
    if os.path.exists(NB_MODEL_PATH):
        return joblib.load(NB_MODEL_PATH)

    model = Pipeline([
        ("count", CountVectorizer(
            ngram_range=(1, 2),
            stop_words=None, 
            min_df=3,
            max_df=0.9,
            max_features=20000
        )),
        ("clf", OneVsRestClassifier(MultinomialNB(alpha=1.0)))
    ])

    model.fit(X_text_clean, y)
    joblib.dump(model, NB_MODEL_PATH)
    return model



MAX_WORDS = 10000
MAX_LEN = 50

def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        words = str(text).lower().split()
        for word in words:
            if word not in vocab and idx < MAX_WORDS:
                vocab[word] = idx
                idx += 1
    return vocab

@st.cache_resource
def prepare_pytorch_data():
    vocab = build_vocab(X_text)

    def encode(text):
        words = str(text).lower().split()
        if len(words) == 0:
            tokens = [vocab["<UNK>"]]
        else:
            tokens = [vocab.get(word, vocab["<UNK>"]) for word in words]

        tokens = tokens[:MAX_LEN]
        tokens += [0] * (MAX_LEN - len(tokens))
        return tokens

    X_encoded = np.array([encode(t) for t in X_text], dtype=np.int64)
    return vocab, torch.tensor(X_encoded, dtype=torch.long), torch.tensor(y)

vocab, X_tensor, y_tensor = prepare_pytorch_data()

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = EmotionDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        return self.fc(h)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])
        return self.fc(h)


def train_torch_model(model, epochs=35):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    model.train()

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        status.write(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")
        progress.progress((epoch + 1) / epochs)

    return model



def load_or_train_gru():
    model = GRUModel(len(vocab), 64, 64, num_labels)

    if os.path.exists(GRU_PATH):
        state = torch.load(GRU_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    model = train_torch_model(model, epochs=35)
    torch.save(model.state_dict(), GRU_PATH)
    model.eval()
    return model

def load_or_train_lstm():
    model = LSTMModel(len(vocab), 64, 64, num_labels)

    if os.path.exists(LSTM_PATH):
        state = torch.load(LSTM_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    model = train_torch_model(model, epochs=35)
    torch.save(model.state_dict(), LSTM_PATH)
    model.eval()
    return model


if model_choice == "Naive Bayes":
    with st.spinner("Loading / training Naive Bayes..."):
        model = train_or_load_naive_bayes()
    st.success("Naive Bayes model ready")
elif model_choice == "GRU":
    with st.spinner("Loading / training GRU..."):
        model = load_or_train_gru()
    st.success("GRU model ready")
else:
    with st.spinner("Loading / training LSTM..."):
        model = load_or_train_lstm()
    st.success("LSTM model ready")



st.subheader("Enter text")
user_text = st.text_area("Text")

threshold = st.slider("Probability threshold", 0.1, 0.9, 0.3)

def encode_single(text):
    words = str(text).lower().split()
    if len(words) == 0:
        tokens = [vocab["<UNK>"]]
    else:
        tokens = [vocab.get(word, vocab["<UNK>"]) for word in words]

    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)

if st.button("Predict") and user_text.strip():

    if model_choice == "Naive Bayes":
        probs = model.predict_proba([clean_text_medium(user_text)])[0]
    else:
        model.eval()
        with torch.no_grad():
            input_tensor = encode_single(user_text)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

    st.subheader("Predicted emotions:")
    found = False
    for emotion, prob in zip(emotion_cols, probs):
        if prob >= threshold:
            st.write(f"{emotion}: {prob:.2f}")
            found = True

    if not found:
        st.write("No emotion above threshold")
