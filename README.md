# sports_betting_optimization
Series of Python scripts to compile data, train ML models, and optimize NFL money line betting.

## Theory and Purpose

### Motivation
The title of the repository may seem unspecific to some; why call this endevour "sports betting" when the scope contains only money lines* for National Football League regular season games? The purpose of the ambiguous title is to convey a preliminary question, which itself begs a more specific question for every answer given: "How do I optimize (x)?" My answer to this general question was "statistics, probably." So I'd like to apply statistical methods (particularly machine learning and neural network models) to optimize... something.

Many of the computationally and mathematically inclined (sometimes called data scientists) flock to financial services and projects in the stock market; but in my studies, I have heard the warning, "past asset data is not directly indicative of future asset data" more times than I can count. This inspires the next question: "What subject's data is most temporally homogeneous?" There is no unique answer to this question, but I settled on sports (specifically the NFL) for a couple reasons: the first of which was my father; apparently a prolific gambler back in the day (and I suppose now that the apple doesn't fall far from the tree). Additionally, my close friends are all sports nuts and I was sick of having nothing to say in these conversations. But a less poetic reason being theoretically why the NFL was most temporally homogeneous compared to other sports and even college football.

* See the _Future Implementations_ section for plans to include other popular bets

### Theory
I theorized it was based on the **total number of _active_ players** and **venerability**. The latter reason is more obvious than the prior; NFL players are quite literally seasoned professionals with many years of experience compared to their collegiate counterparts. Even between other professional sports. I believe the only other sport with a higher average player age is Major League Baseball. This experience and professionalism suggests more consistent performance per player. The more relevant aspect is the total number of _active_ players. Although the MLB has a similar number of players per game (eighteen to the NFL's twenty-two), the word to focus on is _active_. In the MLB, it is primarily pitcher versus batter until the ball is hit, and even then only a few players are _active_ on the play; perhaps five players max actually have contact with the ball. But every play in an NFL game, there are twenty-two players (eleven on each team) performing consistant with their skill level. Each position has an _active_ role distict from the others.

Most interestingly, this further aggregates to the entire team's performance. I enjoy the example of "bad day" scenarios, and it goes like this: say player on the team is "having a bad day" and not performing at their average skill-level (generally speaking). If we assume some kind of normal distribuation on a players skill level, then it is similarly likely that another player on that team is "having a _good_ day." Keep in mind the second reason, these are professionals which should perform consistantly. This translates mathematically to a low variance in the normal distribution we are envisioning. Indeed, some positions are deemed more important than others (e.g. the quarterback). But this again makes NFL the best choice when seeking consistant performance. With our MLB example, if the pitcher is "having a bad day", this much more greatly impacts that team's performace because there are so few _active_ players. But in the NFL, there are ten other players with the opportunity to perform consistantly and (somewhat) independently if the quarterback is performing poorly (or even injured).

### Purpose
The goal of the project is simple: determine the probability of the home team winning for each game per week and use this coupled with moneyline information to optimize the betting strategy. We may ask, "Why only bet on money lines?"* This is primarily because it lessens risk compared to betting on the spread, and it is directly correlated with the results of the neural network model.

There is also one glaring caveat to convince others of my passion for the project itself: **_Sports betting is illegal in the state of Wisconsin_** where I currently reside. This does not stop me from "paper trading" so to say.

* Again, please see the _Future Implementations_ section for plans to include other popular bets

### Results
I proudly have two aspects of my work to share: the model I've constructed is performing with an accuracy of over 65%, and the risk mitigation strategy has allowed for an average return of $180 over ten week of the regular 2021 football season. As to not bamboozle readers, it should be important to note the returns have a standard deviation of about $274. An otherwise very scary statistic, should the overall bankroll not total $1625.

Again, I'm very proud of my work so far. And I look forward to greater collaboration and development to further improve the process for the 2022 football season.

## Surface-level Workflow for Using These Scripts:
1. Collect feature data using collect_feature_data.py.
2. Statistically analyze features using feature_analysis.py.
3. Train neural network using train_FNN.py.
4. Determine optimal betting strategy using risk_analysis.py.

## Future Implementations
1. Analyze the over-under (and possibly the spread, in addition to the money line) using the already implemented functional topology of the neural network. This will open more opportunities to a more diversified betting strategy.
2. Consider additional risk measures. It may not be best to hinge a betting strategy on one risk parameter, as telling as it may be. Note that other parameters have been experimented with (such as expectation, standard deviation, COV, etc), but these have more often led the strategy astray.
