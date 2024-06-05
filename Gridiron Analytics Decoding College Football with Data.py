import os
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import cfbd
from sqlalchemy import create_engine, text

# Set some configurations
pd.set_option('display.max_columns', None)  # Do not truncate columns

# Fetch environment variables
CFBD_API_KEY = os.getenv('CFBD_API_KEY')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Print environment variables for debugging
print("CFBD_API_KEY:", CFBD_API_KEY)
print("DB_USERNAME:", DB_USERNAME)
print("DB_PASSWORD:", DB_PASSWORD)
print("DB_HOST:", DB_HOST)
print("DB_PORT:", DB_PORT)
print("DB_NAME:", DB_NAME)

# Configuration for cfbd API
def configure_cfbd_api(api_key):
    config = cfbd.Configuration()
    config.api_key['Authorization'] = api_key
    config.api_key_prefix['Authorization'] = 'Bearer'
    return cfbd.ApiClient(config)

api_config = configure_cfbd_api(CFBD_API_KEY)

# Initialize APIs
teams_api = cfbd.TeamsApi(api_config)
conferences_api = cfbd.ConferencesApi(api_config)
games_api = cfbd.GamesApi(api_config)
players_api = cfbd.PlayersApi(api_config)
stats_api = cfbd.StatsApi(api_config)
rankings_api = cfbd.RankingsApi(api_config)
recruiting_api = cfbd.RecruitingApi(api_config)
venues_api = cfbd.VenuesApi(api_config)
coaches_api = cfbd.CoachesApi(api_config)

# Fetch data from APIs
def fetch_data(api, fetch_function, **kwargs):
    try:
        data = fetch_function(**kwargs)
        if not data:
            raise ValueError("No data retrieved.")
        return data
    except cfbd.ApiException as e:
        print(f"Error fetching data: {e}")
        return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

# Flatten nested dictionary columns
def flatten_columns(df, col):
    col_flat = pd.json_normalize(df[col])
    col_flat.columns = [f"{col}_{subcol}" for subcol in col_flat.columns]
    df = df.drop(col, axis=1).join(col_flat)
    return df

# Fetch and process all data
teams = fetch_data(teams_api, teams_api.get_fbs_teams)
conferences = fetch_data(conferences_api, conferences_api.get_conferences)
games = fetch_data(games_api, games_api.get_games, year=2021)
player_stats = fetch_data(players_api, players_api.get_player_season_stats, year=2021)
team_stats = fetch_data(stats_api, stats_api.get_team_season_stats, year=2021)
rankings = fetch_data(rankings_api, rankings_api.get_team_rankings, year=2021)
recruiting = fetch_data(recruiting_api, recruiting_api.get_recruiting_players, year=2021)
team_recruiting = fetch_data(recruiting_api, recruiting_api.get_recruiting_teams, year=2021)
venues = fetch_data(venues_api, venues_api.get_venues)
coaches = fetch_data(coaches_api, coaches_api.get_coaches)

# Process data
teams_data = [team.to_dict() for team in teams]
conferences_data = [conf.to_dict() for conf in conferences]
games_data = []
for g in games:
    game_dict = {
        'home_team': g.home_team,
        'home_points': g.home_points,
        'away_team': g.away_team,
        'away_points': g.away_points,
        'venue': g.venue,
        'start_date': g.start_date,
        'conference_game': g.conference_game,
        'neutral_site': g.neutral_site,
        'season': g.season,
        'week': g.week,
    }
    games_data.append(game_dict)

games_df = pd.DataFrame(games_data)
games_df['margin'] = games_df['home_points'] - games_df['away_points']
player_stats_data = [stat.to_dict() for stat in player_stats]
team_stats_data = [stat.to_dict() for stat in team_stats]
rankings_data = []
for ranking in rankings:
    for poll in ranking.polls:
        for rank in poll.ranks:
            rank_dict = rank.to_dict()
            rank_dict.update({
                'season': ranking.season,
                'week': ranking.week,
                'poll': poll.poll,
                'team': rank.school
            })
            rankings_data.append(rank_dict)

rankings_df = pd.DataFrame(rankings_data)
recruiting_data = [rec.to_dict() for rec in recruiting]
team_recruiting_data = [rec.to_dict() for rec in team_recruiting]
venues_data = [venue.to_dict() for venue in venues]
coaches_data = [coach.to_dict() for coach in coaches]

# Flatten nested dictionary columns if they exist
if 'hometown_info' in recruiting_df.columns:
    recruiting_df = flatten_columns(recruiting_df, 'hometown_info')
if 'location' in venues_df.columns:
    venues_df = flatten_columns(venues_df, 'location')
if 'seasons' in coaches_df.columns:
    coaches_df = coaches_df.explode('seasons')
    coaches_df = flatten_columns(coaches_df, 'seasons')

# Database connection
db_string = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db_engine = create_engine(db_string)

# Insert data into PostgreSQL database
def insert_data_to_db(df, table_name, engine):
    df.to_sql(table_name, engine, if_exists='replace', index=False)

insert_data_to_db(games_df, 'cfb_games_2021', db_engine)
insert_data_to_db(teams_df, 'cfb_teams_2021', db_engine)
insert_data_to_db(conferences_df, 'cfb_conferences_2021', db_engine)
insert_data_to_db(player_stats_df, 'cfb_player_stats_2021', db_engine)
insert_data_to_db(team_stats_df, 'cfb_team_stats_2021', db_engine)
insert_data_to_db(rankings_df, 'cfb_rankings_2021', db_engine)
insert_data_to_db(recruiting_df, 'cfb_recruiting_2021', db_engine)
insert_data_to_db(team_recruiting_df, 'cfb_team_recruiting_2021', db_engine)
insert_data_to_db(venues_df, 'cfb_venues_2021', db_engine)
insert_data_to_db(coaches_df, 'cfb_coaches_2021', db_engine)

# Aggregated rankings data to ensure unique team-season combinations
rankings_df = rankings_df.groupby(['team', 'season']).agg({
    'rank': 'min',
    'conference': 'first',
    'first_place_votes': 'sum',
    'points': 'sum'
}).reset_index()

# Merged recruiting data with rankings data on 'team' and 'season'
merged_df = pd.merge(recruiting_df, rankings_df, left_on=['team', 'season'], right_on=['team', 'season'], how='inner')

# Set the season as the time index
merged_df['season'] = pd.to_datetime(merged_df['season'], format='%Y')
merged_df.set_index('season', inplace=True)

# Sort by season
merged_df = merged_df.sort_index()

print(merged_df.head())

# Time Series Analysis and Visualization
# Plotted the time series data for Florida
team_name = 'Florida' 
team_data = merged_df[merged_df['team'] == team_name]

plt.figure(figsize=(12, 6))
plt.plot(team_data.index, team_data['rank_x'], marker='o')
plt.title(f'Time Series of End-of-Season Rankings for {team_name}')
plt.xlabel('Season')
plt.ylabel('Rank')
plt.gca().invert_yaxis()  # Invert y-axis to show rank 1 at the top
plt.grid(True)
plt.show()

# Decomposed the time series data
decomposition = seasonal_decompose(team_data['rank_x'], model='additive', period=1)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# Fitted ARIMA model
model = ARIMA(team_data['rank_x'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecasted future rankings
forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)
print(f'Forecasted Rankings for the next {forecast_steps} seasons:')
print(forecast)

# Fetched the actual rankings for 2022 and 2023 
actual_rankings_2022_2023 = pd.DataFrame({
    'season': pd.to_datetime(['2022-01-01', '2023-01-01']),
    'team': ['Florida', 'Florida'],
    'rank': [74, 84]  # Replace these with actual fetched data
})

# Appended actual rankings to the historical data
historical_and_actual_rankings = pd.concat([merged_df.reset_index(), actual_rankings_2022_2023])

# Set the season as the time index
historical_and_actual_rankings['season'] = pd.to_datetime(historical_and_actual_rankings['season'])
historical_and_actual_rankings.set_index('season', inplace=True)

# Filtered data for Florida
florida_data = historical_and_actual_rankings[historical_and_actual_rankings['team'] == 'Florida']

# Trained the SARIMAX model on the historical data (2018-2021)
train_data = florida_data.loc['2018-01-01':'2021-01-01']['rank_x']
model = SARIMAX(train_data, order=(1, 1, 1))
results = model.fit()

# Forecasted the rankings for 2022 and 2023
forecast = results.get_forecast(steps=2)
forecast_index = pd.date_range(start='2022-01-01', periods=2, freq='AS-JAN')
forecast_df = forecast.summary_frame()
forecast_df.index = forecast_index
forecast_df['team'] = 'Florida'

# Combined actual and forecasted data for comparison
combined_df = pd.concat([florida_data.reset_index(), forecast_df.reset_index()])
combined_df = combined_df.rename(columns={'index': 'season', 'rank_x': 'actual_rank', 'mean': 'forecasted_rank'})

# Plotted the actual vs forecasted rankings
plt.figure(figsize=(12, 6))
plt.plot(combined_df['season'], combined_df['actual_rank'], marker='o', label='Actual Rank', color='blue')
plt.plot(combined_df['season'], combined_df['forecasted_rank'], marker='x', linestyle='--', label='Forecasted Rank', color='red')
plt.title('Comparison of Actual vs Forecasted End-of-Season Rankings for Florida')
plt.xlabel('Season')
plt.ylabel('Rank')
plt.legend()
plt.grid()
plt.show()

# Printed the comparison data
print(combined_df[['season', 'team', 'actual_rank', 'forecasted_rank']])

actual_rankings_2022_2023 = pd.DataFrame({
    'season': pd.to_datetime(['2022-01-01', '2023-01-01']),
    'team': ['Florida', 'Florida'],
    'actual_rank': [74, 84]  # Replace with actual fetched data
})

# Inspected the columns for duplicates
print("Columns in combined_df:", combined_df.columns)
print("Columns in actual_rankings_2022_2023:", actual_rankings_2022_2023.columns)

# Removed duplicate columns in combined_df
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Ensured the season column is properly formatted as datetime in both DataFrames
combined_df.loc[:, 'season'] = pd.to_datetime(combined_df['season'], errors='coerce')
actual_rankings_2022_2023.loc[:, 'season'] = pd.to_datetime(actual_rankings_2022_2023['season'], errors='coerce')

# Ensured unique indices in both DataFrames
combined_df = combined_df.reset_index(drop=True).drop_duplicates(subset=['season']).set_index('season')
actual_rankings_2022_2023 = actual_rankings_2022_2023.reset_index(drop=True).drop_duplicates(subset=['season']).set_index('season')

# Merged actual rankings with forecasted data
comparison_df = pd.concat([combined_df, actual_rankings_2022_2023[['actual_rank']]], axis=1)

# Plotted actual vs forecasted rankings
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['rank'], marker='o', label='Actual Rank (2018-2021)', color='blue')
plt.plot(comparison_df.index, comparison_df['forecasted_rank'], marker='x', linestyle='--', label='Forecasted Rank (2022-2023)', color='red')
plt.plot(comparison_df.index, comparison_df['actual_rank'], marker='s', linestyle='-', label='Actual Rank (2022-2023)', color='green')
plt.title('Comparison of Actual vs Forecasted End-of-Season Rankings for Florida')
plt.xlabel('Season')
plt.ylabel('Rank')
plt.legend()
plt.grid()
plt.show()

# Printed the comparison data
print(comparison_df[['team', 'rank', 'forecasted_rank', 'actual_rank']])

#Go Gators!!
#By Shaun McKellar Jr
