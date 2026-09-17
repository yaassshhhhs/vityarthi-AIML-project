# Project Statement — AI-Based Badminton Shot Prediction System

## Problem Statement

In competitive badminton, choosing the right shot — a **Smash**, **Drop**, or **Clear** —
is a split-second decision that depends on multiple fast-changing factors: where the
player and opponent are standing on the court, how difficult the incoming shuttle is to
play, and how much pressure the current score situation creates. Beginner and
intermediate players often struggle to internalize this decision-making process, and
there is no lightweight, accessible tool that lets a player experiment with different
game situations and instantly see which shot a data-driven model would recommend and why.

This project addresses that gap by building a machine learning system that takes a
snapshot of a rally situation (player position, opponent position, score pressure, and
shuttle difficulty) and predicts the most appropriate shot to play, along with the
model's confidence in each possible option.

## Scope of the Project

- Generate a **synthetic, rule-based dataset** of 300 realistic badminton rally
  scenarios (player/opponent court coordinates, score situation, shuttle difficulty,
  and the resulting recommended shot).
- Perform **exploratory data analysis (EDA)** to understand class distribution and
  feature relationships.
- Train and evaluate a **Decision Tree Classifier** (scikit-learn) that predicts one of
  three shot classes: `Smash`, `Drop`, `Clear`.
- Persist the trained model and the label encoders so they can be reused without
  retraining.
- Provide an **interactive Streamlit web application** where a user can set match
  conditions using sliders and dropdowns, visualize player/opponent positions on a
  scaled court diagram, and receive a shot recommendation with confidence scores and a
  short tactical explanation.
- The project is a **decision-support and educational tool**, not a real-time computer
  vision or sensor-based system — it does not track a live match; all inputs are
  entered manually by the user.
- Out of scope: live video/sensor tracking of players and shuttles, multi-shot
  rally/sequence prediction, doubles-match logic, and mobile app deployment.

## Target Users

- **Beginner and intermediate badminton players** who want to understand which shot
  suits a given court situation and build better shot-selection intuition.
- **Badminton coaches and trainers** who want a simple visual aid to explain shot
  selection logic to students during training sessions.
- **AI/ML students and educators** who want a compact, end-to-end example of a
  classification pipeline — from synthetic data generation to model training,
  evaluation, and deployment behind a simple web UI.

## High-Level Features

| # | Feature | Description |
|---|---------|--------------|
| 1 | Synthetic Dataset Generator | `generate_dataset.py` creates 300 rule-based badminton scenarios with realistic shot-selection logic. |
| 2 | Exploratory Data Analysis | Notebook cells visualize shot distribution, shuttle difficulty vs. shot type, score situation vs. shot type, and player-position vs. shot-type scatter plots. |
| 3 | Machine Learning Model | A `DecisionTreeClassifier` (max depth 6) is trained on an 80/20 train-test split to classify the recommended shot. |
| 4 | Model Evaluation | Accuracy, precision/recall/F1 (classification report), confusion matrix, and feature-importance analysis. |
| 5 | Model Persistence | The trained model and the three `LabelEncoder` objects are saved with `joblib` for reuse by the web app. |
| 6 | Interactive Web App | A Streamlit app lets users set player/opponent positions (sliders), score situation, and shuttle difficulty (dropdowns) and click "Predict Best Shot". |
| 7 | Court Visualization | A live matplotlib court diagram plots the player and opponent positions relative to the net. |
| 8 | Prediction Explanation | The app shows the recommended shot, a plain-language tactical tip, and a confidence-score breakdown across all three shot classes. |
