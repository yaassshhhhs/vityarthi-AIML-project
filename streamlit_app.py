import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Badminton Shot Predictor",
    page_icon="🏸",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .shot-box {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .smash  { background: linear-gradient(135deg,#ff6b6b,#ee5a24); color: white; }
    .drop   { background: linear-gradient(135deg,#4ecdc4,#0abde3); color: white; }
    .clear  { background: linear-gradient(135deg,#45b7d1,#2e86de); color: white; }
    .info-card {
        background: #f8f9fa;
        border-left: 5px solid #0abde3;
        padding: 1rem 1.4rem;
        border-radius: 8px;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODEL_DIR, "badminton_model.pkl"))
    le_score   = joblib.load(os.path.join(MODEL_DIR, "le_score.pkl"))
    le_shuttle = joblib.load(os.path.join(MODEL_DIR, "le_shuttle.pkl"))
    le_shot    = joblib.load(os.path.join(MODEL_DIR, "le_shot.pkl"))
    return model, le_score, le_shuttle, le_shot

try:
    model, le_score, le_shuttle, le_shot = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found. Please run the notebook first to train & save the model.\n\n`{e}`")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏸 Badminton Shot Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered shot recommendation using a Decision Tree classifier</div>',
            unsafe_allow_html=True)
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🎯 Enter Game Conditions")

    with st.expander("📍 Player & Opponent Positions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Your Position**")
            player_x = st.slider("Player X (court width)", 0.0, 10.0, 5.0, 0.1)
            player_y = st.slider("Player Y (court length)", 0.0, 13.4, 11.0, 0.1)
        with c2:
            st.markdown("**Opponent Position**")
            opponent_x = st.slider("Opponent X", 0.0, 10.0, 5.0, 0.1)
            opponent_y = st.slider("Opponent Y", 0.0, 13.4, 2.0, 0.1)

    with st.expander("📊 Match Conditions", expanded=True):
        score_situation = st.selectbox(
            "Score Situation",
            ["Early", "Mid", "Pressure"],
            help="Early = beginning of game | Mid = ongoing | Pressure = deciding points"
        )
        shuttle_difficulty = st.selectbox(
            "Shuttle Difficulty",
            ["Easy", "Medium", "Hard"],
            help="Based on shuttle height, speed, and reachability"
        )

    predict_btn = st.button("🔮 Predict Best Shot", use_container_width=True, type="primary")

# ── Court visualisation ───────────────────────────────────────────────────────
with col_result:
    st.subheader("🏟️ Court View & Prediction")

    # Draw court
    fig, ax = plt.subplots(figsize=(4.5, 6))
    fig.patch.set_facecolor('#e8f5e9')
    ax.set_facecolor('#c8e6c9')

    court = mpatches.FancyBboxPatch((0, 0), 10, 13.4,
                                     boxstyle="round,pad=0.1",
                                     linewidth=2, edgecolor='darkgreen', facecolor='#a5d6a7')
    ax.add_patch(court)
    ax.axhline(y=6.7, color='white', linewidth=2.5, linestyle='-', label='Net', zorder=3)
    # Service lines
    ax.axhline(y=1.98, color='white', linewidth=1, linestyle='--', alpha=0.6)
    ax.axhline(y=11.42, color='white', linewidth=1, linestyle='--', alpha=0.6)

    # Player marker
    ax.scatter([player_x], [player_y], c='#ff6b6b', s=200, zorder=5, edgecolors='white', linewidths=2)
    ax.annotate('You', (player_x, player_y), textcoords="offset points",
                xytext=(8, 6), fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='#e74c3c', alpha=0.85))

    # Opponent marker
    ax.scatter([opponent_x], [opponent_y], c='#45b7d1', s=200, zorder=5, edgecolors='white', linewidths=2)
    ax.annotate('Opp', (opponent_x, opponent_y), textcoords="offset points",
                xytext=(8, 6), fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='#2e86de', alpha=0.85))

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 14.5)
    ax.set_xlabel('Court Width →', fontsize=9)
    ax.set_ylabel('Court Length →', fontsize=9)
    ax.set_title('Court Positions', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── Prediction result ─────────────────────────────────────────────────────
    if predict_btn and model_loaded:
        ss_enc = le_score.transform([score_situation])[0]
        sd_enc = le_shuttle.transform([shuttle_difficulty])[0]

        input_df = pd.DataFrame([{
            'Player_X': player_x,
            'Player_Y': player_y,
            'Opponent_X': opponent_x,
            'Opponent_Y': opponent_y,
            'Score_Situation': ss_enc,
            'Shuttle_Difficulty': sd_enc,
        }])

        pred_enc = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        shot = le_shot.inverse_transform([pred_enc])[0]
        proba_dict = dict(zip(le_shot.classes_, pred_proba))

        css_class = shot.lower()
        emoji = {"Smash": "💥", "Drop": "🎯", "Clear": "🛡️"}.get(shot, "🏸")
        description = {
            "Smash": "Aggressive downward strike — use when you have a high shuttle and opponent is out of position.",
            "Drop": "Soft shot near the net — use to force opponent forward and create deception.",
            "Clear": "High defensive shot to baseline — use to reset play or buy recovery time.",
        }.get(shot, "")

        st.markdown(f"""
        <div class="shot-box {css_class}">
            {emoji} Recommended Shot: {shot.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="info-card">📘 <b>Strategy:</b> {description}</div>',
                    unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### 📊 Confidence Scores")
        bar_colors = {"Clear": "#45b7d1", "Drop": "#4ecdc4", "Smash": "#ff6b6b"}
        for s, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
            st.metric(label=s, value=f"{p*100:.1f}%")
            st.progress(float(p))

    elif predict_btn and not model_loaded:
        st.warning("Model not loaded. Please train the model first.")

# ── About section ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About This Project"):
    st.markdown("""
**AI-Based Badminton Shot Prediction System**
- **Algorithm**: Decision Tree Classifier (scikit-learn)
- **Dataset**: 300 synthetic samples with realistic badminton game logic
- **Features**: Player/Opponent XY positions, Score Situation, Shuttle Difficulty
- **Output**: One of — Smash 💥 | Drop 🎯 | Clear 🛡️

Built as a BYOP project for an AI/ML course. The model acts as a decision-support
tool for beginner and intermediate badminton players.
    """)
