"""
Sports Betting Project: Get Money Lines
Author: Trevor Cross
Last Updated: 01/20/22

Scrape money lines for NFL games from Draftkngs sportsbook.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import scraping libraries
import requests
from bs4 import BeautifulSoup
import re

# -----------------------------
# ---Define Primary Function---
# -----------------------------

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
    soup = str(BeautifulSoup(requests.get(url), 'html.parser'))
    
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
        matchups_formatted.append(matchup.replace("'","").replace(" ",""))
        
    moneylines_df = pd.DataFrame(data=moneyline_pairs, 
                        index=matchups_formatted, 
                        columns=['moneyline_home', 'moneyline_away'])
    
    return moneylines_df

# ----------------
# ---Run Script---
# ----------------

if __name__ == "__main__":
    
    url = "https://sportsbook.draftkings.com/leagues/football/88670561"
    
    moneylines_df = scrape_moneylines(url)