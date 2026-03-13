# Predictive Keyboard using PyTorch and Streamlit

This project is a simple **next-word prediction system** built using **PyTorch** and **Streamlit**. It works like a basic predictive keyboard by suggesting the most likely next words based on a short text prompt entered by the user.

The model is trained on **Sherlock Holmes stories** and uses an **LSTM (Long Short-Term Memory)** network for sequence modeling.

---

## Features

- Word-level next-word prediction
- LSTM-based deep learning model
- Top-k next word suggestions
- Interactive Streamlit web app
- Built with PyTorch
- Handles unknown words using `<UNK>` token

---

## Project Structure

```bash
predictive-keyboard/
│
├── data/
│   └── sherlock-holm.es_stories_plain-text_advs.txt
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
