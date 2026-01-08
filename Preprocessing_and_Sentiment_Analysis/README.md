# Preprocessing and Sentiment Inference

This folder contains scripts and notebooks used for parsing, preprocessing,
and sentiment inference on Google Trends related-query data.

Model training details (hyperparameters, epochs) follow the configuration reported in the manuscript.

## Parsing and Merging

The notebook `parse_merge-clean-queries.ipynb` implements:
- parsing of downloaded CSV files,
- merging of monthly datasets into a unified structure, and
- alignment of temporal and brand identifiers.

These steps ensure a consistent input format for downstream analysis.

## Text Preprocessing

Also text normalization and preprocessing are implemented in `parse_merge-clean-queries.ipynb`
using the Hazm library, which is specifically designed for Persian language processing.

Preprocessing steps include:
- character normalization,
- removal of extraneous whitespace and symbols, and
- standardization of orthographic variants.

These steps reduce noise and improve robustness of downstream sentiment inference.

## Sentiment Inference

Text sentiment_analysis is implemented in `hybrid_sentiment_analysis.py` and trained model save in `hybrid_sentiment_model.h5` and auto-labeling script implemented in`auto_labele_related_queries.py`
Sentiment inference is performed using a hybrid "Lexicon + FastText  + CNN + BiLSTM"  neural architecture.
The model assigns polarity labels (positive or negative) to related search queries.

Model characteristics:
- Pre-trained Persian FastText embeddings,
- CNN layers for local semantic pattern extraction,
- BiLSTM layers for contextual sequence modeling.

Neutral queries are excluded from model training but retained for descriptive aggregation.

The output of this pipeline consists of annual counts of positive and negative queries per brand,
which are subsequently used for EKPI construction.

## Output Indicators

The sentiment pipeline produces annual counts of:
- Positive queries (PS),
- Negative queries (NS).

These outputs are used to construct sentiment-based EKPIs in later stages.

