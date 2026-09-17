# 🏸 AI-Based Badminton Shot Prediction System

> A machine learning project that predicts the most suitable badminton shot — **Smash**, **Drop**, or **Clear** — based on real-time gameplay conditions.

---

## 📌 Project Overview

In competitive badminton, shot selection is a critical split-second decision influenced by:
- Player and opponent positions on the court
- Match pressure (score situation)
- Shuttle difficulty (height, speed, reachability)

This project trains a **Decision Tree Classifier** on a synthetic dataset of realistic game scenarios to recommend the optimal shot type. It also includes a **Streamlit web app** for interactive predictions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Model | Decision Tree Classifier (scikit-learn) |
| 📊 Dataset | 300 synthetic, rule-based badminton scenarios |
| 🎯 Prediction | Smash / Drop / Clear shot recommendation |
| 📈 Evaluation | Accuracy, F1-score, Confusion Matrix, Feature Importance |
| 🌐 Web App | Interactive Streamlit UI with court visualisation |
| 📓 Notebook | Step-by-step Jupyter Notebook |

---

## 🗂️ Project Structure

```
badminton_shot_predictor/
│
├── data/
│   ├── generate_dataset.py       # Script to generate synthetic dataset
│   └── badminton_dataset.csv     # Generated dataset (300 rows)
│
├── notebooks/
│   └── Badminton_Shot_Prediction.ipynb   # Full ML pipeline notebook
│
├── model/
│   ├── badminton_model.pkl       # Trained Decision Tree model
│   ├── le_score.pkl              # LabelEncoder for Score_Situation
│   ├── le_shuttle.pkl            # LabelEncoder for Shuttle_Difficulty
│   └── le_shot.pkl               # LabelEncoder for Recommended_Shot
│
├── app/
│   └── streamlit_app.py          # Interactive Streamlit web app
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🛠️ Installation

### 1. Clone / Download the Repository

```bash
git clone https://github.com/yaassshhhhs/vityarthi-AIML-project.git
cd badminton-shot-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Generate the Dataset

```bash
cd data
python generate_dataset.py
cd ..
```

### Step 2: Run the Jupyter Notebook (Training & Evaluation)

```bash
jupyter notebook notebooks/Badminton_Shot_Prediction.ipynb
```

Run all cells — this will train the model, evaluate it, and save the `.pkl` files to the `model/` directory.

### Step 3: Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## 📷 Example Usage

**Streamlit App — Input:**
- Player Position: X=5.0, Y=12.0 (back of court)
- Opponent Position: X=5.0, Y=1.5 (near net)
- Score Situation: Pressure
- Shuttle Difficulty: Easy

**Output:**  
`💥 Recommended Shot: SMASH` — Confidence: 87%

---

## 📦 Dependencies (`requirements.txt`)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
jupyter
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Decision Tree Classifier |
| Max Depth | 6 |
| Min Samples Split | 5 |
| Min Samples Leaf | 3 |
| Train / Test Split | 80% / 20% |
| Random State | 42 |

**Why Decision Tree?**
- Easy to interpret (important for coaching context)
- Handles mixed feature types natively
- No feature scaling required
- Reveals which game factors matter most

---

## 👨‍💻 Author
**Name - Yash Pratap Singh**
--
**Reg no. - 25MIM10193**
--
AI/ML Course 
--
**THANK YOU**

---

