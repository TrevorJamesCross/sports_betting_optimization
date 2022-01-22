"""
Sports Betting Project: Toolbox
Author: Trevor Cross
Last Updated: 01/22/22

Library of functions used to support Sports Betting Project. More specifically, 
most functions are used in the risk_analysis.py script.
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

# import scraping libraries
import requests
from bs4 import BeautifulSoup
import re

# import support libraries
from tqdm import tqdm
from time import time
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# ---Define Primary Functions---
# ------------------------------

def collect_team_stats(current_year, current_week):
    """
    Parameters
    ----------
    current_year : integer
        NFL season from which to collect team stats from.
    current_week : integer
        The week in the season to collect data up to.
        
    Returns
    -------
    stats_df : pandas.DataFrame
        DataFrame containing up-to-date statistics for each team this season.
    """
    
    # define team stats df; resets every season
    stats_df = pd.DataFrame(index=[team.abbreviation for team in Teams(current_year)],
                            columns=['avg_pts_for','avg_pts_let',
                                     'avg_rsh','avg_pss',
                                     'pss_rate','3rd_dn_rate',
                                     '4th_dn_rate','pss_td',
                                     'rsh_td','1st_dns',
                                     'sacks','trnovrs',
                                     'pens','pen_yrds']).fillna(0.0)
            
    # define team totals df; resets every season
    tots_df = pd.DataFrame(index=[team.abbreviation for team in Teams(current_year)],
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
    boxscores_year = Boxscores(1,current_year,current_week-1).games
    
    for week_num, week_key in enumerate( tqdm(boxscores_year, desc='Getting team stats', unit='week') ):
        for game_num, game in enumerate( boxscores_year[week_key] ):
            while True:
                try:
                    # get team abbreviations
                    home_abbr, away_abbr = (game['home_abbr'].upper(),
                                            game['away_abbr'].upper())
     
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
        for abbr in [team.abbreviation for team in Teams(current_year)]:
            stats_df.loc[abbr][['pss_td','rsh_td','1st_dns','sacks','trnovrs','pens','pen_yrds']] = (
                tots_df.loc[abbr][['pss_td','rsh_td','1st_dns','sacks','trnovrs','pens','pen_yrds']] )
            
            stats_df.loc[abbr]['avg_pts_for'] = tots_df.loc[abbr]['pts_for'] / (week_num + 1)
            stats_df.loc[abbr]['avg_pts_let'] = tots_df.loc[abbr]['pts_let'] / (week_num + 1)
            
            stats_df.loc[abbr]['avg_rsh'] = tots_df.loc[abbr]['rsh_yrds'] / tots_df.loc[abbr]['rsh_att']
            
            stats_df.loc[abbr]['avg_pss'] = tots_df.loc[abbr]['pss_yrds'] / tots_df.loc[abbr]['pss_att']
            stats_df.loc[abbr]['pss_rate'] = tots_df.loc[abbr]['pss_comp'] / tots_df.loc[abbr]['pss_att']
            
            stats_df.loc[abbr]['3rd_dn_rate'] = tots_df.loc[abbr]['3rd_dn_con'] / tots_df.loc[abbr]['3rd_dn_att']
            stats_df.loc[abbr]['4th_dn_rate'] = tots_df.loc[abbr]['4th_dn_con'] / tots_df.loc[abbr]['4th_dn_att']
            
    return stats_df



def scrape_moneylines(url):
    """
    Parameters
    ----------
    url : string
        url to Draftkings NFL sportsbook.

    Returns
    -------
    moneylines_df : pandas.DataFrame
        DataFrame indexed by team matchups this week [home,away], with data
        containing their respective moneylines
    """
    
    print("Getting money lines... This may take several minutes...")
    
    # define team dictionary (INEXHAUSTIVE)
    team_dict = {
        "CIN Bengals" : "CIN",
        "TEN Titans" : "OTI",
        "SF 49ers" : "SFO",
        "GB Packers" : "GNB",
        "LA Rams" : "RAM",
        "TB Buccaneers" : "TAM",
        "BUF Bills" : "BUF",
        "KC Chiefs" : "KAN"}
    
    # get html string from Draftkings
    soup = str(BeautifulSoup(requests.get(url).content, 'html.parser'))
    
    # get teams
    teams = re.findall('name-text">(.*?)</div>', soup)
    teams = [team_dict[team] for team in teams]
    
    # get moneylines
    moneylines = re.findall('<span class="sportsbook-odds american no-margin default-color">(.*?)</span>', soup)
    
    # create matchups & format them
    matchups = []
    moneyline_pairs = []
    for i in range(0,len(teams),2):
        matchups.append(str([teams[i+1],teams[i]]))
        moneyline_pairs.append([moneylines[i+1],moneylines[i]])
    
    matchups_formatted = []
    for matchup in matchups:
        matchups_formatted.append(matchup.replace("'","").replace(" ","").replace("[","").replace("]",""))
        
    moneylines_df = pd.DataFrame(data=moneyline_pairs, 
                        index=matchups_formatted, 
                        columns=['moneyline_home', 'moneyline_away'])
    
    return moneylines_df



def construct_risk_reward(mls_pred_df):
    """
    Parameters
    ----------
    mls_pred_df : pandas.DataFrame
        Contains home and away moneylines, and the percent chance of a home
        victory per matchup index.

    Returns
    -------
    risk_reward_df : pandas.DataFrame
        Contains amount of money needed to reach the money line to bet on 
        favored team, the amount to be returned if bet is successful, and the 
        probability that the bet is successful.
    """
    
    # define risk_reward_df
    risk_reward_df = pd.DataFrame(index=mls_pred_df.index,
                                  columns=['risk','reward','prob'])
    # iterate through all matchups
    for matchup in mls_pred_df.index:
        
        # determine if home team is expected to win
        if mls_pred_df.loc[matchup]['home_vict'] >= 0.50:
            
            # determine if bet is on favorite
            if mls_pred_df.loc[matchup]['moneyline_home'] < 0.0:
                risk_reward_df.loc[matchup]['risk'] = abs(mls_pred_df.loc[matchup]['moneyline_home'])
                risk_reward_df.loc[matchup]['reward'] = abs(mls_pred_df.loc[matchup]['moneyline_home']) + 100
                risk_reward_df.loc[matchup]['prob'] = mls_pred_df.loc[matchup]['home_vict']
                
            # or on underdog 
            else:
                risk_reward_df.loc[matchup]['risk'] = 100
                risk_reward_df.loc[matchup]['reward'] = mls_pred_df.loc[matchup]['moneyline_home'] + 100
                risk_reward_df.loc[matchup]['prob'] = mls_pred_df.loc[matchup]['home_vict']
                
        # or if away team is expected to win        
        else:
            
            # determine if bet is on favorite
            if mls_pred_df.loc[matchup]['moneyline_away'] < 0.0:
                risk_reward_df.loc[matchup]['risk'] = abs(mls_pred_df.loc[matchup]['moneyline_away'])
                risk_reward_df.loc[matchup]['reward'] = abs(mls_pred_df.loc[matchup]['moneyline_away']) + 100
                risk_reward_df.loc[matchup]['prob'] = 1 - mls_pred_df.loc[matchup]['home_vict']
                
            # or on underdog 
            else:
                risk_reward_df.loc[matchup]['risk'] = 100
                risk_reward_df.loc[matchup]['reward'] = mls_pred_df.loc[matchup]['moneyline_away'] + 100
                risk_reward_df.loc[matchup]['prob'] = 1 - mls_pred_df.loc[matchup]['home_vict']
                
    return risk_reward_df



def get_all_combos(risk_reward_df, min_bets, max_bets):
    """
    Parameters
    ----------
    risk_reward_df : pandas.DataFrame
        Contains amount of money needed to reach the money line to bet on 
        favored team, the amount to be returned if bet is successful, and the 
        probability that the bet is successful.
    min_bets : Integer
        Describes the minimum number of bets to be placed.
    max_bets : Integer
        Describes the maximum number of bets to be placed.

    Returns
    -------
    combo_prob_pairs : list
        Contains all possible betting combinations and their respective chance
        to make money.
    """
    
    # define combo, prob (of making money) pairs
    combo_prob_pairs = []
    
    # define number of bets range
    num_bets_range = np.arange(min_bets, max_bets+1)
    
    for num_bets in num_bets_range:
        for sub_matchups in list( combinations(risk_reward_df.index, num_bets) ):
            
            # define prob of making money for combo
            prob_in_black = 0
            
            # convert tuple to list
            sub_matchups = list(sub_matchups)
            
            # define number of wins range (we could lose none or all games)
            num_wins_range = np.arange(0, len(risk_reward_df.loc[sub_matchups])+1)
            
            for num_wins in num_wins_range:
                for won_matchups in list( combinations(risk_reward_df.loc[sub_matchups].index, num_wins)):
                    
                    # conver tuple to list
                    won_matchups = list(won_matchups)
                    
                    # get lost matchups
                    lost_matchups = [matchup for matchup in risk_reward_df.loc[sub_matchups].index if matchup not in won_matchups]
                    
                    # calculate net profit
                    net_prof = ( sum(risk_reward_df.loc[won_matchups]['reward'] - risk_reward_df.loc[won_matchups]['risk']) -
                                sum(risk_reward_df.loc[lost_matchups]['risk']) )
                    
                    # add to prob if made money
                    if net_prof > 0:
                        prob_in_black += ( np.prod(risk_reward_df.loc[won_matchups]['prob']) *
                                          np.prod(1 - risk_reward_df.loc[lost_matchups]['prob']) )
                        
            # append combo df with respective prob of making money 
            combo_prob_pairs.append([risk_reward_df.loc[sub_matchups], prob_in_black])
                
    return combo_prob_pairs