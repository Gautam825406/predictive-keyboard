import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

MODEL_PATH = "predictive_keyboard_model.pth"
VOCAB_PATH = "vocab_data.pth"

st.set_page_config(
    page_title="Predictive Keyboard AI",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def predict_next_words(model, word2idx, idx2word, sequence_length, device, text_prompt, top_k=3):
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) < sequence_length - 1:
        raise ValueError(f"Please enter at least {sequence_length - 1} words.")

    input_seq = tokens[-(sequence_length - 1):]
    input_ids = [word2idx.get(word, word2idx["<UNK>"]) for word in input_seq]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, top_k)

    suggestions = [
        {"word": idx2word[idx.item()], "prob": prob.item()}
        for prob, idx in zip(top_probs, top_indices)
    ]
    return suggestions


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
        color: white;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(168,85,247,0.25));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }

    .chip {
        display: inline-block;
        padding: 0.7rem 1.1rem;
        margin: 0.3rem 0.4rem 0.3rem 0;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #f8fafc;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        margin-top: 2rem;
        font-size: 0.95rem;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: rgba(255,255,255,0.06) !important;
        color: white !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        font-size: 1.05rem !important;
        padding: 14px !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: rgba(255,255,255,0.06) !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.7rem 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }

    div.stButton > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">⌨️ Predictive Keyboard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Beautiful next-word prediction app built with <b>PyTorch</b> + <b>Streamlit</b></div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("About")
    st.write(
        "This app predicts the most likely next words from a sentence prompt using a trained LSTM language model."
    )
    st.markdown("---")
    top_k = st.slider("Number of suggestions", 1, 10, 3)
    st.markdown("---")
    st.caption("Make sure `train.py` has been run before launching this app.")

# -----------------------------
# Load model
# -----------------------------
try:
    model, word2idx, idx2word, sequence_length, device = load_assets()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value">Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Context Needed</div>
                <div class="metric-value">{sequence_length - 1} words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Device</div>
                <div class="metric-value">{str(device).upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter your prompt</div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "",
            height=140,
            placeholder="Type at least a few words...\nExample: so are we really at",
            label_visibility="collapsed",
        )

        predict_btn = st.button("✨ Generate Suggestions")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick examples</div>', unsafe_allow_html=True)

        examples = [
            "so are we really at",
            "he looked at the",
            "i do not know",
            "it was a very",
        ]
        for ex in examples:
            st.code(ex, language=None)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Prediction results
    # -----------------------------
    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                suggestions = predict_next_words(
                    model, word2idx, idx2word, sequence_length, device, user_input, top_k
                )

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Top Suggestions</div>', unsafe_allow_html=True)

                chip_html = ""
                for item in suggestions:
                    chip_html += f'<span class="chip">{item["word"]}</span>'
                st.markdown(chip_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">Confidence Scores</div>', unsafe_allow_html=True)

                for item in suggestions:
                    st.write(f"**{item['word']}** — {item['prob'] * 100:.2f}%")
                    st.progress(min(float(item["prob"]), 1.0))

                st.markdown("</div>", unsafe_allow_html=True)

                # Bonus preview
                best_word = suggestions[0]["word"]
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Autocomplete Preview</div>', unsafe_allow_html=True)
                st.write(f"**{user_input} {best_word}**")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(str(e))

    st.markdown(
        '<div class="footer-note">Built with Streamlit • PyTorch • NLP</div>',
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(str(e))
    st.info("Run `python train.py` first to generate the model files.")