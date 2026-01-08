# EKPI and Composite Index Calculation (DAX Documentation)

This document describes the logic used to construct External Key Performance Indicators (EKPIs)
and the composite Brand Position in Google (BPG) index.

All calculations and transformations described below were implemented using DAX formulas
within Microsoft Power BI.
---

## Role of DAX in the Framework

DAX was selected because:
- It enables transparent aggregation over large time-indexed datasets,
- It integrates naturally with interactive visualization workflows,
- It allows consistent implementation of weighted composite indicators.

Executable Power BI files are not distributed due to platform dependency.

---

## EKPI Construction Logic

EKPIs are constructed through annual aggregation of monthly indicators:

- Volume-based indicators (e.g., SOVG) are computed as yearly sums of monthly values.
- Growth indicators (e.g., SOVGR, BAGR) are computed using the compound annual growth rate (CAGR).
- Sentiment indicators aggregate annual counts of inferred positive and negative queries.

---

## Normalization Strategy

All EKPIs are normalized to a unified **1–100 scale** immediately prior to composite aggregation.

This normalization:
- Ensures compatibility with expert-elicited weights obtained via the Best–Worst Method (BWM),
- Avoids zero-valued criteria that could nullify weighted contributions,
- Reduces instability caused by negative or near-zero growth values.

The appropriate stage for normalization is specified in the following sections (Sections of EKPIs Calculation).

Note: No additional normalization is required in the initial stages. However, normalizing the data after extraction is optional for visual understanding, but this has no effect on the final EKPI result and only imposes unnecessary overhead on the framework.
---

## SOVG and SOVGR Calculation

Annual SOVG values were obtained by aggregating monthly values within each calendar year.

Annual growth rates (SOVGR) were computed using the compound annual growth rate (CAGR)
applied to the corresponding monthly series.

Since the resulting annual indicators exhibited heterogeneous ranges,
 Normalization to the 1–100 scale was applied before composite aggregation.

## BAG and BAGR Calculation

Google Trends timeline data were collected using a single multi-brand query.

Annual Brand Awareness in Google (BAG) values were obtained by aggregating monthly timeline values within each year.
Annual growth rates (BAGR) were computed using the CAGR formulation.

As with SOVG and SOVGR, final BAG and BAGR values were normalized to a 1–100 scale
to ensure consistency across all EKPIs.

## Sentiment-Based Indicators (PS, NS, PSR) Calculation

Sentiment-based indicators were constructed by aggregating the annual counts of
positive (PS) and negative (NS) queries inferred by the sentiment model.

Net sentiment (PS–NS) and positive sentiment ratio (PSR = PS / (PS + NS))
were computed at the annual level.

Final sentiment indicators were normalized to the 1–100 scale prior to BPG aggregation.

## Composite EKPI (BPG) Calculation

The Brand Position in Google (BPG) index is computed as a weighted linear aggregation of normalized EKPIs.

Weights are derived using the Best–Worst Method and applied consistently across brands and time.
