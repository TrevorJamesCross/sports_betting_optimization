"""
Sports Betting Project: Collect Feature Data
Author: Trevor Cross
Last Updated: 01/30/22

Collect NFL game data and engineer differential features for neural network
training. Data is gathered using the Sports Reference API, sportsipy.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import sportsipy libraries
from sportsipy.nfl.boxscore import Boxscore, Boxscores
from sportsipy.nfl.teams import Teams

# import support functions
from tqdm import tqdm
from time import time
import warnings
warnings.filterwarnings("ignore")

# import request library
from requests.exceptions import ConnectionError

# -----------------------------
# ---Define Primary Function---
# -----------------------------

def collect_nfl_data(seasons):
    """
    Parameters
    ----------
    seasons : iterable
        Years to collect data from.

    Returns
    -------
    final_df : pandas.DataFrame
        Feature data to be used when training machine learning models.
    """
    
    # define final_df
    final_df = pd.DataFrame()
    
    # iterate through selected seasons
    for year in tqdm(seasons, desc='Collecting data', unit='season'):
        
        # define team stats df; resets every season
        stats_df = pd.DataFrame(index=[team.abbreviation for team in Teams(year)],
                                columns=['avg_pts_for','avg_pts_let',
                                         'avg_rsh','avg_pss',
                                         'pss_rate','3rd_dn_rate',
                                         '4th_dn_rate','pss_td',
                                         'rsh_td','1st_dns',
                                         'sacks','trnovrs',
                                         'pens','pen_yrds']).fillna(0.0)
        
        # define team totals df; resets every season
        tots_df = pd.DataFrame(index=[team.abbreviation for team in Teams(year)],
                               columns=['pts_for','pts_let',
                                        'rsh_yrds','pss_yrds',
                                        'rsh_att','pss_att',
                                        'rsh_td','pss_td',
                                        '1st_dns','pss_comp',
                                        'sacks','trnovrs',
                                        'pens','pen_yrds',
                                        '3rd_dn_att','4th_dn_att',
                                        '3rd_dn_con','4th_dn_con']).fillna(0.0)
        
        # get boxscore info
        boxscores_year = Boxscores(1,year,20).games

        for week_num, week_key in enumerate( boxscores_year ):
            for game_num, game in enumerate( boxscores_year[week_key] ):
                while True:
                    try:
                        # get team abbreviations
                        home_abbr, away_abbr = (game['home_abbr'].upper(),
                                                game['away_abbr'].upper())
                        
                        # concatenate differential data after the fourth week
                        if week_num > 3:
                            home_vict = int(game['home_abbr'] == game['winning_abbr'])
                            
                            diffs_df = pd.concat([ stats_df.loc[home_abbr] - stats_df.loc[away_abbr],
                                                  pd.Series(home_vict) ])
                            
                            final_df = final_df.append(diffs_df, ignore_index=True)
         
                        # get game info
                        info = Boxscore(game['boxscore'])
                        
                        # get team stats
                        home_stats = [info.home_points, info.away_points,
                                      info.home_rush_yards, info.home_pass_yards,
                                      info.home_rush_attempts, info.home_pass_attempts,
                                      info.home_rush_touchdowns, info.home_pass_touchdowns,
                                      info.home_first_downs, info.home_pass_completions,
                                      info.away_times_sacked, info.away_turnovers,
                                      info.home_penalties, info.home_yards_from_penalties,
                                      info.home_third_down_attempts, info.home_fourth_down_attempts,
                                      info.home_third_down_conversions, info.home_fourth_down_conversions]
                        
                        away_stats = [info.away_points, info.home_points,
                                      info.away_rush_yards, info.away_pass_yards,
                                      info.away_rush_attempts, info.away_pass_attempts,
                                      info.away_rush_touchdowns, info.away_pass_touchdowns,
                                      info.away_first_downs, info.away_pass_completions,
                                      info.home_times_sacked, info.home_turnovers,
                                      info.away_penalties, info.away_yards_from_penalties,
                                      info.away_third_down_attempts, info.away_fourth_down_attempts,
                                      info.away_third_down_conversions, info.away_fourth_down_conversions]
                        
                        # update team_tots_df
                        try:
                            tots_df.loc[home_abbr] = tots_df.loc[home_abbr] + home_stats
                            tots_df.loc[away_abbr] = tots_df.loc[away_abbr] + away_stats
                        except TypeError as exc:
                            raise UserWarning("Selected seasons may not have required data; terminating script... ") from exc
                            
                        
                    # catch connection error; retries with same game
                    except ConnectionError:
                        print(">>> Connection lost; retrying... ")
                        time.sleep(5)
                        continue
                    
                    # break while loop if succesful; moves to next game
                    break
                
            # construct stats_df from tots_df
            for abbr in [team.abbreviation for team in Teams(year)]:
                stats_df.loc[abbr][['pss_td','rsh_td','1st_dns','sacks','trnovrs','pens','pen_yrds']] = (
                    tots_df.loc[abbr][['pss_td','rsh_td','1st_dns','sacks','trnovrs','pens','pen_yrds']] )
                
                stats_df.loc[abbr]['avg_pts_for'] = tots_df.loc[abbr]['pts_for'] / (week_num + 1)
                stats_df.loc[abbr]['avg_pts_let'] = tots_df.loc[abbr]['pts_let'] / (week_num + 1)
                
                stats_df.loc[abbr]['avg_rsh'] = tots_df.loc[abbr]['rsh_yrds'] / tots_df.loc[abbr]['rsh_att']
                
                stats_df.loc[abbr]['avg_pss'] = tots_df.loc[abbr]['pss_yrds'] / tots_df.loc[abbr]['pss_att']
                stats_df.loc[abbr]['pss_rate'] = tots_df.loc[abbr]['pss_comp'] / tots_df.loc[abbr]['pss_att']
                
                stats_df.loc[abbr]['3rd_dn_rate'] = tots_df.loc[abbr]['3rd_dn_con'] / tots_df.loc[abbr]['3rd_dn_att']
                stats_df.loc[abbr]['4th_dn_rate'] = tots_df.loc[abbr]['4th_dn_con'] / tots_df.loc[abbr]['4th_dn_att']
    
    return final_df.fillna(0.0).rename(columns={0:"home_win"})

# ----------------
# ---Run Script---
# ----------------

if __name__ == "__main__":
    
    seasons = np.arange(2000,2021)
    feature_data = collect_nfl_data(seasons)