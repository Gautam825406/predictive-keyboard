import os
import random
from collections import Counter

import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Download tokenizer resources
# -----------------------------
nltk.download("punkt")

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/sherlock-holm.es_stories_plain-text_advs.txt"
SEQUENCE_LENGTH = 4   # total window size => 3 input words + 1 target word
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SAMPLES = 30000   # reduce for faster training
MODEL_PATH = "predictive_keyboard_model.pth"
VOCAB_PATH = "vocab_data.pth"
SEED = 42

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# Load and tokenize text
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = word_tokenize(text)
print(f"Total tokens: {len(tokens)}")


# -----------------------------
# Build vocabulary
# -----------------------------
special_tokens = ["<PAD>", "<UNK>"]
word_counts = Counter(tokens)
vocab_words = sorted(word_counts, key=word_counts.get, reverse=True)
vocab = special_tokens + vocab_words

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")


def encode_word(word):
    return word2idx.get(word, word2idx["<UNK>"])


# -----------------------------
# Create sequences
# -----------------------------
data = []
for i in range(len(tokens) - SEQUENCE_LENGTH):
    input_seq = tokens[i:i + SEQUENCE_LENGTH - 1]
    target = tokens[i + SEQUENCE_LENGTH - 1]
    data.append((input_seq, target))

random.shuffle(data)
data = data[:MAX_SAMPLES]

print(f"Training samples used: {len(data)}")


# -----------------------------
# Dataset class
# -----------------------------
class NextWordDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        input_ids = torch.tensor([encode_word(w) for w in input_seq], dtype=torch.long)
        target_id = torch.tensor(encode_word(target), dtype=torch.long)
        return input_ids, target_id


dataset = NextWordDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------------
# Model
# -----------------------------
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, _ = self.lstm(x)            # (batch, seq_len, hidden_dim)
        output = self.fc(output[:, -1, :])  # use last time step
        return output


model = PredictiveKeyboard(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# Train
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        loss.backward()

        # gradient clipping helps LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")


# -----------------------------
# Save model + vocab
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
torch.save(
    {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "sequence_length": SEQUENCE_LENGTH,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "vocab_size": vocab_size,
    },
    VOCAB_PATH,
)

print(f"Model saved to {MODEL_PATH}")
print(f"Vocab/config saved to {VOCAB_PATH}")