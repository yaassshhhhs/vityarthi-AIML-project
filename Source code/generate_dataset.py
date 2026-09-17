import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

n = 300

data = []

for _ in range(n):
    player_x = round(random.uniform(0.0, 10.0), 2)
    player_y = round(random.uniform(0.0, 13.4), 2)
    opponent_x = round(random.uniform(0.0, 10.0), 2)
    opponent_y = round(random.uniform(0.0, 13.4), 2)
    score_situation = random.choice(["Early", "Mid", "Pressure"])
    shuttle_difficulty = random.choice(["Easy", "Medium", "Hard"])

    # Rule-based shot logic (realistic badminton strategy)
    # Distance from net (player_y close to 0 = near net on their side)
    near_net = player_y < 4.0
    at_back = player_y > 9.5
    center = 4.0 <= player_y <= 9.5

    # Opponent positioning
    opp_near_net = opponent_y > 9.5   # opponent on far net side
    opp_at_back = opponent_y < 4.0    # opponent at back
    opp_center = 4.0 <= opponent_y <= 9.5

    if shuttle_difficulty == "Easy" and at_back and opp_near_net:
        shot = "Smash"
    elif shuttle_difficulty == "Easy" and at_back and opp_center:
        shot = "Smash" if random.random() > 0.3 else "Clear"
    elif shuttle_difficulty == "Hard" and at_back:
        shot = "Clear"
    elif shuttle_difficulty == "Hard" and center:
        shot = "Clear" if random.random() > 0.3 else "Drop"
    elif shuttle_difficulty == "Medium" and near_net:
        shot = "Drop"
    elif shuttle_difficulty == "Easy" and near_net:
        shot = "Drop" if random.random() > 0.2 else "Smash"
    elif score_situation == "Pressure" and shuttle_difficulty == "Hard":
        shot = "Clear"
    elif score_situation == "Pressure" and shuttle_difficulty == "Easy":
        shot = "Smash"
    elif score_situation == "Early" and shuttle_difficulty == "Medium":
        shot = random.choice(["Drop", "Clear"])
    elif opp_at_back and shuttle_difficulty in ["Easy", "Medium"]:
        shot = "Drop"
    elif shuttle_difficulty == "Medium" and at_back:
        shot = "Clear" if random.random() > 0.4 else "Smash"
    else:
        shot = random.choice(["Smash", "Drop", "Clear"])

    data.append({
        "Player_X": player_x,
        "Player_Y": player_y,
        "Opponent_X": opponent_x,
        "Opponent_Y": opponent_y,
        "Score_Situation": score_situation,
        "Shuttle_Difficulty": shuttle_difficulty,
        "Recommended_Shot": shot
    })

df = pd.DataFrame(data)
df.to_csv("badminton_dataset.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(df["Recommended_Shot"].value_counts())
print(df.head())
