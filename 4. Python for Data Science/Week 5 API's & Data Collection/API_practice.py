
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# from nba_api.stats.static import teams
# from nba_api.stats.endpoints import leaguegamefinder

# # dict_={'a':[11,21,31],'b':[12,22,32]}
# # df=pd.DataFrame(dict_)

# filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Labs/Golden_State.pkl"
# file_name = "Golden_State.pkl"

# def download(url, filename):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(filename, "wb") as f:
#             f.write(response.content)

# def one_dict(list_dict):
#     keys=list_dict[0].keys()
#     out_dict={key:[] for key in keys}
#     for dict_ in list_dict:
#         for key, value in dict_.items():
#             out_dict[key].append(value)
#     return out_dict

# nba_teams = teams.get_teams()
# # nba_teams2 = nba_teams[0:3]
# dict_nba_team = one_dict(nba_teams)
# df_teams = pd.DataFrame(dict_nba_team)
# df_warriors = df_teams[df_teams['nickname']=='Warriors']
# id_warriors = df_warriors[['id']].values[0][0]
# gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id_warriors)
# gf = gamefinder.get_json()
# games = gamefinder.get_data_frames()[0]
# games2 = pd.read_pickle(file_name)
# games_home=games[games['MATCHUP']=='GSW vs. TOR']
# games_away=games[games['MATCHUP']=='GSW @ TOR']
# gamesh = games_home['PLUS_MINUS'].mean()
# gamesa = games[games['MATCHUP']=='GSW @ TOR']
# dload = download(filename, "Golden_State.pkl")
# mean = games_home["PTS"].mean()
# mean2 = games_away["PTS"].mean()

# fig, ax = plt.subplots()
# games_away.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
# games_home.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
# ax.legend(["away", "home"])
# # plot = plt.show()

# # print(df)
# # print(df.mean())
# # print(df_teams[0:3])
# # print(df_teams)
# # print(df_warriors)
# # print(gf)
# # print(games.head())
# # print(dload)
# # print(games2.head())
# # print(gamesh)
# # print(gamesa)
# # print(plot)

# print(mean)
# print(mean2)

#  PART 2

# from randomuser import RandomUser
import pandas as pd

# r = RandomUser()
# some_list = r.generate_users(10)
# name = r.get_full_name()

# # for user in some_list:
# #     print (user.get_full_name()," ",user.get_email())

# # for user in some_list:
# #     print (user.get_picture())

# def get_users():
#     users =[]
     
#     for user in RandomUser.generate_users(10):
#         users.append({"Name":user.get_full_name(),"Gender":user.get_gender(),"City":user.get_city(),"State":user.get_state(),"Email":user.get_email(), "DOB":user.get_dob(),"Picture":user.get_picture()})
      
#     return pd.DataFrame(users) 

# df1 = pd.DataFrame(get_users())

# # print(some_list)
# # print(get_users())
# print(df1)

# PART 3

import requests
import json

# data = requests.get("https://fruityvice.com/api/fruit/all")
# results = json.loads(data.text)
# pd.DataFrame(results)
# df2 = pd.json_normalize(results)
# cherry = df2.loc[df2["name"] == 'Cherry']
# (cherry.iloc[0]['family']) , (cherry.iloc[0]['genus'])
# banana = df2.loc[df2["name"] == "Banana"] 
# (banana.iloc[0]['family']) , (banana.iloc[0]['genus'])

# print(df2)
# print(cherry)
# print(banana)

url = 'https://official-joke-api.appspot.com/jokes/ten'
r = requests.get(url)

results = json.loads(r.text)
dj = pd.json_normalize(results)
dj.drop(columns=["type","id"],inplace=True)
# results
print(dj)