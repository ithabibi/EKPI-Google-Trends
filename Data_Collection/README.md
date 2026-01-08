# Data Acquisition Notes

This document describes the data acquisition procedures used in the study
“A Framework for Developing External Key Performance Indicators Using Google Trends and Multi-Criteria Decision-Making”.

This file documents the procedures used to collect all raw data required for EKPI construction.

The focus of this documentation is transparency and methodological reproducibility, rather than automation.

---

## Scope

This document focuses on the collection of external digital trace data from:
- Google Search (search index counts), and
- Google Trends (timeline and related queries),

in compliance with platform policies and terms of service.

---

## General Principles

All data were collected:
- Using synchronized queries across competing brands,
- Within identical temporal windows,
- In compliance with Google platform policies and terms of service.

Manual data collection was intentionally adopted where automated access was restricted.

---

## A1: Google Search Index Counts (SOVG Inputs)

Monthly Google Search index counts were collected manually for each brand using identical query formulations.

Key characteristics:
- Queries were executed for all brands within the same session.
- Monthly values were recorded consistently over the full observation period.
- Data were stored in CSV format and archived on Mendeley Data.
- Columns name: index,date,keyword,search_result

These values serve as inputs for annual Share of Voice in Google (SOVG) and its growth rate (SOVGR).

---

## A2: Google Trends Timeline Data (BAG Inputs)

Brand awareness signals were collected using Google Trends timeline queries.

To ensure cross-brand comparability:
- All brands were retrieved simultaneously within a single multi-term query.
- Google Trends’ internal normalization (0–100) therefore applies uniformly across brands.

The resulting monthly timelines were exported in CSV format and used directly for annual aggregation.

---

## A3: Google Trends Related Queries (Sentiment Inputs)

Related search queries were collected monthly for each brand using Google Trends.

These queries represent user-generated intent expressions and form the textual corpus for sentiment inference.

All related-query collections were synchronized across brands and time to preserve comparability.

---

## Archiving and Availability

All representative raw datasets are archived on Mendeley Data with permanent DOIs.
Direct redistribution of platform interfaces or dashboards is intentionally avoided.
