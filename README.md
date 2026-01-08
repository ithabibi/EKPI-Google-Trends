

# EKPI Framework Based on Google Trends

This repository documents the computational and analytical pipeline supporting the study:

“A Framework for Developing External Key Performance Indicators Using Google Trends and Multi-Criteria Decision-Making”.

The purpose of this repository is to ensure methodological transparency, traceability, and conceptual reproducibility of the proposed EKPI framework, rather than to distribute executable dashboards or proprietary platform files.

This repository emphasizes conceptual and methodological reproducibility rather than platform-specific executability.
---

## Research Scope

The study proposes a structured framework for constructing External Key Performance Indicators (EKPIs) by integrating:
- Google Search index volumes,
- Google Trends awareness signals,
- Sentiment signals inferred from related search queries.

These indicators are aggregated into a composite index termed **Brand Position in Google (BPG)** using expert-elicited weights obtained via the Best–Worst Method (BWM).

---

## Repository Design Philosophy

The repository follows three design principles:

1. **Policy compliance**  
   Data acquisition procedures respect platform policies and therefore avoid automated scraping or redistribution of proprietary interfaces.

2. **Conceptual reproducibility**  
   All transformations, aggregation logic, and indicator definitions are fully documented, even when platform-dependent tools (e.g., Power BI DAX) are used.

3. **Traceability to the process model**  
   Each repository component corresponds directly to one or more stages in the methodological process model (Figure 4 in the manuscript).

---

## Repository Structure

- `Data_Collection/`  
  Documentation of manual data collection procedures for Google Search and Google Trends data, designed to comply with platform policies and terms of service.

- `Preprocessing_and_Sentiment_Analysis/`  
  Python scripts and notebooks for parsing and merging downloaded datasets, text normalization for Persian language data, and sentiment inference using a hybrid "Lexicon+FastText+CNN+BiLSTM" model.

- `EKPI_Development/`  
  Conceptual documentation of EKPI and BPG construction logic, including full documentation of DAX formulas used for indicator calculation and aggregation in Microsoft Power BI.

- `docs/`  
  Mapping between repository components and the process model presented in Figure 4 of the manuscript.

---

## Reproducibility Statement

All core analytical steps are implemented using open-source Python scripts or presented with detailed documentation and are fully reproducible.

Platform-dependent computations (e.g., DAX formulas executed within Microsoft Power BI) are documented for transparency but are not provided as executable files to avoid dependencies on proprietary software.

---

## Data Availability

Representative datasets used in this study are archived on Mendeley Data with permanent DOIs, as reported in the Data Availability section of the manuscript.

## Citation

If you use this repository or its documentation, please cite the associated article.


