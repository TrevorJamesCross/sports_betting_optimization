"""
Sports Betting Project: Risk Analysis
Author: Trevor Cross
Last Updated: 01/30/22

Get team stats and construct differential data for the week. Run this through
the neural network prediction to obtain a percent chance of home victory. Given
the prediciton and money lines, optimize the betting strategy by calculating 
the probability of a net profit at or above zero dollars.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import ML libraries
import joblib 
from tensorflow.keras.models import load_model

# import support functions
from operator import itemgetter
from toolbox import scrape_moneylines, collect_team_stats, construct_risk_reward, get_all_combos

# ------------------------
# ---Scrape Money Lines---
# ------------------------

# define Draftkings url
url = "https://sportsbook.draftkings.com/leagues/football/88670561"

# get moneylines (and matchups)
moneylines_df = scrape_moneylines(url)

# -----------------------------
# ---Build Differential Data---
# -----------------------------

# define season and week
current_year = 2021
current_week = 22

# get team stats
team_stats_df = collect_team_stats(current_year, current_week)

# remove last feature
team_stats_df = team_stats_df.drop(columns=team_stats_df.columns[-1])

# build differential df
diffs_df = pd.DataFrame(index=moneylines_df.index, columns=team_stats_df.columns).fillna(0.0)
for matchup in moneylines_df.index:
    home_abbr, away_abbr = matchup.split(",")
    diffs_df.loc[matchup] = team_stats_df.loc[home_abbr] - team_stats_df.loc[away_abbr]

# normalize and reshape differential data
scaler_path = "/home/tjcross/sports_betting_optimization/saved_scaler"
scaler = joblib.load(scaler_path)

diffs_data = scaler.transform(diffs_df.to_numpy()).reshape(diffs_df.shape[0],1,diffs_df.shape[1])

# ---------------------
# ---Get predictions---
# ---------------------

# define neural network model path
model_path = "/home/tjcross/sports_betting_optimization/saved_model"

# load saved model & automatically compile
model = load_model(model_path, compile=True)

# make predictions
preds_df = pd.DataFrame(data=model.predict(diffs_data).reshape(diffs_data.shape[0]),
                        index=moneylines_df.index,
                        columns=['home_vict'])

# concatenate moneylines and predictions
mls_preds_df = pd.concat([moneylines_df,preds_df], axis=1).astype(float)

# -----------------------
# ---Run Risk Analysis---
# -----------------------

# construct risk-reward df
risk_reward_df = construct_risk_reward(mls_preds_df).astype(float)

# get all betting combos and their probs of making money (sorted by prob of success)
combo_prob_pairs = sorted(get_all_combos(risk_reward_df, 1, len(risk_reward_df)), 
                          key=itemgetter(1),
                          reverse=True)