# Mapping Between Repository Components and Figure 4

This document maps the components of the process model presented in Figure 4 of the manuscript
to the corresponding materials in this repository.

## Component A – Data Acquisition
- Folder: `Data_Collection/`
- Description: Manual data collection procedures.

## Component B – Preprocessing
- Folder: `Preprocessing_and_Sentiment_Analysis/`
- Files: `parse_merge-clean-queries.ipynb`
- Description: Parse, merge, and clean all downloaded csv files from the Google Trends query pipeline.

## Component C – Sentiment Analysis
- Folder: `Preprocessing_and_Sentiment_Analysis/`
- Files: Sentiment analysis script `hybrid_sentiment_analysis.py`, the trained saved model `hybrid_sentiment_model.h5`, and auto-labeling script `auto_label_related_queries.py`
- Description: Implement hybrid (Lexicon + FastText  + CNN + BiLSTM) sentiment analysis pipeline.

## Component D – EKPI Development
- Folder: `EKPI_Development/`
- Description: Indicator construction logic and DAX-based aggregation.

## End-to-End Traceability Summary

Each component of the Figure 4 process model is mapped to either:
- Executable open-source code (Components B and C), or
- Fully documented manual or platform-dependent procedures (Components A and D).

This design ensures methodological transparency and traceability while respecting platform constraints and proprietary software boundaries.

