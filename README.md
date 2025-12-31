# Daily Goal Dynamics: Similarity, Momentum, and Affect

This repository contains the analysis notebooks used to explore daily goal-setting behavior in a multi-month daily diary study. The goal of this work is to understand how goal similarity across days interacts with goal completion, effort, and affect, and to distinguish goal momentum from goal continuation effects.

This repository is intended for internal review and advisor feedback. Analyses are exploratory and robustness-focused.

---

## Repository Structure

data/
- raw/                  Original diary exports (not shared publicly)
- proc/            Cleaned, analysis-ready datasets

notebooks (TBD)/

environment.yml         Conda environment for reproducibility

---

## Research Questions

1. Descriptive patterns of goal types, effort, completion, and affect
2. Associations between goal type distributions and academic grades
3. Alignment between daily and monthly goals across the month
4. Relationships between affect and goal completion
5. Momentum vs compensation in daily goal pursuit
6. Role of multi-day continuation goals
7. Affect as a potential mechanism

---

## Dataset Overview

Participants: ~183 total (final analytic sample: 112)
Duration: Two cohorts, approximately three months each
Total days: ~7,800 after preprocessing
Goals per day: Two

Goals were entered on the evening of day t for execution on day t+1. All goals were shifted forward by one day to align with completion, effort, and affect measures.

---

## Goal Representation

Each goal was categorized into one of 35 goal types using GPT-4.

Goal similarity between today and tomorrow was computed in two ways:
- Binary similarity using GPT-4 same-goal classification
- Continuous similarity using cosine similarity of OpenAI text embeddings

Daily-to-monthly similarity was computed as the maximum similarity between daily and monthly goals.

---

## Affect Measures

Positive affect consisted of 8 items.
Negative affect consisted of 12 items.

Both scales showed high internal consistency:
- Positive affect alpha approximately 0.89
- Negative affect alpha approximately 0.90

Composite scores were created by averaging items within each scale.

---

## Modeling Strategy

All inferential analyses use linear mixed-effects models with participant-level random intercepts.

Primary models:
completion(t+1) ~ similarity × completion(t)

Affect models:
completion(t+1) ~ similarity × completion(t) + affect(t+1)

---

## Continuation vs Non-Continuation Goals

To address the possibility that similarity reflects multi-day continuation goals, continuation goals were conservatively defined using:
- GPT same-goal label
- High similarity greater than 0.80
- Low completion (below participant-specific 90th percentile)

All main analyses were re-estimated after removing these cases. Additional robustness checks were conducted at similarity thresholds below 0.70, 0.60, and 0.50.

Continuation splits are treated as sensitivity analyses, not causal partitions.

---

## Interpretation Notes

- Analyses are observational and exploratory

---

## Reproducibility

Create the environment using:

conda env create -f environment.yml
conda activate goal-dynamics

Raw data are not included for privacy reasons.