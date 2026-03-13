import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

# -----------------------------
# Setup
# -----------------------------
nltk.download("punkt")

MODEL_PATH = "predictive_keyboard_model.pth"
VOCAB_PATH = "vocab_data.pth"

st.set_page_config(page_title="Predictive Keyboard", page_icon="⌨️", layout="centered")


# -----------------------------
# Load saved vocab/config
# -----------------------------
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Please run train.py first."
        )

    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            f"Vocab/config file not found: {VOCAB_PATH}. Please run train.py first."
        )

    saved_data = torch.load(VOCAB_PATH, map_location=device)

    word2idx = saved_data["word2idx"]
    idx2word = saved_data["idx2word"]
    sequence_length = saved_data["sequence_length"]
    embed_dim = saved_data["embed_dim"]
    hidden_dim = saved_data["hidden_dim"]
    vocab_size = saved_data["vocab_size"]

    class PredictiveKeyboard(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=word2idx["<PAD>"]
            )
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            output, _ = self.lstm(x)
            output = self.fc(output[:, -1, :])
            return output

    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, word2idx, idx2word, sequence_length, device


def suggest_next_words(model, word2idx, idx2word, sequence_length, device, text_prompt, top_k=3):
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) < sequence_length - 1:
        raise ValueError(f"Please enter at least {sequence_length - 1} words.")

    input_seq = tokens[-(sequence_length - 1):]
    input_ids = [word2idx.get(word, word2idx["<UNK>"]) for word in input_seq]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze(0)
        top_indices = torch.topk(probs, top_k).indices.tolist()

    suggestions = [idx2word[idx] for idx in top_indices]
    return suggestions


# -----------------------------
# UI
# -----------------------------
st.title("⌨️ Predictive Keyboard Web App")
st.write("Type a phrase and get the top 3 predicted next words.")

try:
    model, word2idx, idx2word, sequence_length, device = load_assets()

    st.success("Model loaded successfully.")

    user_input = st.text_input(
        f"Enter at least {sequence_length - 1} words:",
        placeholder="Example: so are we really at"
    )

    top_k = st.slider("Number of suggestions", min_value=1, max_value=10, value=3)

    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                suggestions = suggest_next_words(
                    model, word2idx, idx2word, sequence_length, device, user_input, top_k
                )

                st.subheader("Suggestions")
                cols = st.columns(len(suggestions))
                for i, word in enumerate(suggestions):
                    cols[i].button(word, key=f"suggestion_{i}")

            except Exception as e:
                st.error(str(e))

except Exception as e:
    st.error(str(e))
    st.info("Make sure you have already trained the model by running: python train.py")