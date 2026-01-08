# ============================================================
# Auto-tagging Script for Full Related Queries Dataset
# Uses pre-trained Hybrid CNN+BiLSTM model (Fig. 4 pipeline)
# ============================================================

import pandas as pd
import numpy as np
import hazm
import fasttext
from keras.models import load_model
import keras.backend as K

# -------------------------------
# Load pre-trained model
# -------------------------------
MODEL_PATH = "hybrid_sentiment_model.h5"
model = load_model(MODEL_PATH, compile=False)
print("Loaded pre-trained hybrid model...")

# -------------------------------
# Load full Related Queries
# -------------------------------
csv_related = pd.read_csv("parsed-merged_cleaned_queries.csv")  # full dataset
queries = list(csv_related['Query'])
print("Loaded Related Queries...")

# -------------------------------
# Load FastText embeddings
# -------------------------------
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz
fasttext_model = fasttext.load_model("cc.fa.300.bin")
print("Loaded FastText embeddings...")

# -------------------------------
# Preprocessing functions
# -------------------------------
def CleanPersianText(text):
    normalizer = hazm.Normalizer()
    return normalizer.normalize(text)

def tokenize_and_vectorize(text, max_vocab_token=5, embedding_dim=300):
    x = np.zeros((max_vocab_token, embedding_dim), dtype=K.floatx())
    tokens = hazm.word_tokenize(CleanPersianText(text))
    for i, tok in enumerate(tokens[:max_vocab_token]):
        if tok in fasttext_model.words:
            x[i, :] = fasttext_model.get_word_vector(tok)
    return x

# -------------------------------
# Vectorize all queries
# -------------------------------
embedding_dim = 300
max_vocab_token = 5

x_data = np.zeros((len(queries), max_vocab_token, embedding_dim), dtype=K.floatx())
for idx, q in enumerate(queries):
    x_data[idx] = tokenize_and_vectorize(q)

# -------------------------------
# Predict sentiment
# -------------------------------
y_pred_prob = model.predict(x_data)
y_pred_class = np.argmax(y_pred_prob, axis=1)  # 0=Negative,1=Positive

# Map numeric class to label
label_map = {0: "Negative", 1: "Positive"}
y_pred_label = [label_map[c] for c in y_pred_class]

# -------------------------------
# Save results
# -------------------------------
csv_related['Predicted_Label'] = y_pred_label
csv_related.to_csv("auto_labeled_related_queries.csv", index=False, encoding='utf-8-sig')
print("Auto-labeled Related Queries saved.")
