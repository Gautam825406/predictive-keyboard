import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

MODEL_PATH = "predictive_keyboard_model.pth"
VOCAB_PATH = "vocab_data.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load vocab/config
# -----------------------------
saved_data = torch.load(VOCAB_PATH, map_location=device)

word2idx = saved_data["word2idx"]
idx2word = saved_data["idx2word"]
sequence_length = saved_data["sequence_length"]
embed_dim = saved_data["embed_dim"]
hidden_dim = saved_data["hidden_dim"]
vocab_size = saved_data["vocab_size"]


def encode_word(word):
    return word2idx.get(word, word2idx["<UNK>"])


# -----------------------------
# Model definition
# -----------------------------
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


# -----------------------------
# Load model
# -----------------------------
model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def suggest_next_words(text_prompt, top_k=3):
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) < sequence_length - 1:
        raise ValueError(f"Please enter at least {sequence_length - 1} words.")

    input_seq = tokens[-(sequence_length - 1):]
    input_ids = [encode_word(word) for word in input_seq]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze(0)
        top_indices = torch.topk(probs, top_k).indices.tolist()

    suggestions = [idx2word[idx] for idx in top_indices]
    return suggestions


if __name__ == "__main__":
    print("Predictive Keyboard Ready")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "exit":
            break

        try:
            suggestions = suggest_next_words(user_input, top_k=3)
            print("Suggestions:", suggestions)
        except Exception as e:
            print("Error:", e)