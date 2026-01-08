# ============================================================
# Hybrid "Lexicon + FastText  + CNN + BiLSTM" Sentiment Classification Pipeline
# (Canonical pipeline used in the paper – Fig. 4)
# ============================================================

# -------- Metrics --------
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# -------- Core Libraries --------
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.models import Sequential
import keras.backend as K

# -------- Utilities --------
import numpy as np
import pandas
import random
import hazm
import fasttext
import os
import sys
import codecs
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================================
# Reproducibility (best-effort)
# ============================================================
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ============================================================
# Step 1) Load FastText Persian Embeddings (C2 sub-component)
# ============================================================
# Download once from:
# https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.bin.gz
print("Loading FastText model...")
fasttext_model = fasttext.load_model("cc.fa.300.bin")

# ============================================================
# Step 2) Load Datasets
# ============================================================
# Related Queries (manually labeled subset for training/testing)
csv_related = pandas.read_csv("unique_related_queries_manually_labeled.csv")

# Sentiment Lexicon (used in hybrid training)
csv_lexicon = pandas.read_csv("lexicon-dictionary.csv")

# ============================================================
# Text Normalization (B2 sub-component)
# ============================================================
def CleanPersianText(text):
    normalizer = hazm.Normalizer()
    return normalizer.normalize(text)

# ============================================================
# Prepare Related Queries (B2 sub-component)
# ============================================================
revlist_related = list(map(lambda x: [CleanPersianText(x[0]), x[1]],
                           zip(csv_related['Query'], csv_related['Suggestion'])))

positive_rel = list(filter(lambda x: x[1] == 1, revlist_related))
negative_rel = list(filter(lambda x: x[1] == 2, revlist_related))

# Balance positive and negative samples
positive_rel = positive_rel[:len(negative_rel)]
negative_rel = negative_rel[:len(negative_rel)]

revlist_rel_balanced = positive_rel + negative_rel
random.shuffle(revlist_rel_balanced)

# Split train/test (80/20)
train_size = int(0.80 * len(revlist_rel_balanced))
train_related = revlist_rel_balanced[:train_size]
test_related = revlist_rel_balanced[train_size:]

# ============================================================
# Prepare Lexicon Data (Hybrid uses lexicon in training) (C1 sub-component)
# ============================================================
revlist_lexicon = list(map(lambda x: [CleanPersianText(x[0]), x[1]],
                           zip(csv_lexicon['Query'], csv_lexicon['Suggestion'])))

lex_pos = list(filter(lambda x: x[1] == 1, revlist_lexicon))
lex_neg = list(filter(lambda x: x[1] == 2, revlist_lexicon))

# Merge related queries + lexicon for training (HYBRID)
revlist_train = train_related + lex_pos + lex_neg
random.shuffle(revlist_train)

# ============================================================
# Step 3) Vectorization Parameters (UNCHANGED) (C2 sub-component)
# ============================================================
embedding_dim = 300
max_vocab_token = 5

x_train = np.zeros((len(revlist_train), max_vocab_token, embedding_dim), dtype=K.floatx())
y_train = np.zeros((len(revlist_train), 2), dtype=np.int32)

x_test = np.zeros((len(test_related), max_vocab_token, embedding_dim), dtype=K.floatx())
y_test = np.zeros((len(test_related), 2), dtype=np.int32)

# ============================================================
# Fill Training Data (C3 sub-component)
# ============================================================
for idx, (text, label) in enumerate(revlist_train):
    tokens = hazm.word_tokenize(text)
    for i, tok in enumerate(tokens[:max_vocab_token]):
        if tok in fasttext_model.words:
            x_train[idx, i, :] = fasttext_model.get_word_vector(tok)
    # Encoding: Negative = [1,0], Positive = [0,1]
    y_train[idx, :] = [1.0, 0.0] if label == 2 else [0.0, 1.0]

# ============================================================
# Fill Test Data (Unseen Related Queries)
# ============================================================
for idx, (text, label) in enumerate(test_related):
    tokens = hazm.word_tokenize(text)
    for i, tok in enumerate(tokens[:max_vocab_token]):
        if tok in fasttext_model.words:
            x_test[idx, i, :] = fasttext_model.get_word_vector(tok)
    y_test[idx, :] = [1.0, 0.0] if label == 2 else [0.0, 1.0]

# ============================================================
# Step 4) Build Hybrid CNN + BiLSTM Model (UNCHANGED)
# ============================================================
precision_metric = Precision()
recall_metric = Recall()

model = Sequential()

# -------- CNN layers --------
model.add(Conv1D(32, kernel_size=8, activation='elu',
                 padding='same', input_shape=(max_vocab_token, embedding_dim)))
model.add(Conv1D(64, kernel_size=6, activation='elu', padding='same'))
model.add(Conv1D(96, kernel_size=4, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=4))

# -------- BiLSTM layer --------
model.add(Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.2)))

# -------- Dense layers --------
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile (UNCHANGED)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001, decay=1e-6),
    metrics=['accuracy', 'AUC', precision_metric, recall_metric]
)

# ============================================================
# Step 5) Train Model
# ============================================================
batch_size = 64
epochs = 50

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_data=(x_test, y_test),
    verbose=0
)

# ============================================================
# Step 6) Evaluation
# ============================================================
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
precision = precision_score(y_true, y_pred, average=None, zero_division=0)
recall = recall_score(y_true, y_pred, average=None, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Precision [neg, pos]:", precision)
print("Recall    [neg, pos]:", recall)
print("Confusion Matrix:\n", cm)

# ============================================================
# Step 7) Save Trained Model
# ============================================================
model.save("hybrid_sentiment_model.h5")
print("Hybrid model saved successfully.")
