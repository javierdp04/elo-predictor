import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

COLUMNS_TO_DROP_GAMES = [
                            'site', 'black_rating_diff', 
                            'white_rating_diff', 'white_title', 
                            'black_title', 'eco', 
                            'opening', 'event',
                            'utc_date', 'utc_time',
                            'white', 'black'
                            'time_control'
]

COLUMNS_TO_DROP_MOVES = [
                            'uci', 'san',
                            'fen'
]
SUPERIOR_LIMIT_NUM_PLIES = 200
INFERIOR_LIMIT_NUM_PLIES = 10
SUPERIOR_LIMIT_RATING_DIFF = 75


def clean_games_data(game_data : pd.DataFrame):
    game_data = game_data.copy()
    return( game_data.pipe(filter_by_equal_conditional_expression, 'event', 'Rated Rapid game')
            .pipe(filter_games_with_same_white_and_black_player)
            .pipe(filter_extreme_num_plies_values)
            .pipe(filter_only_interesting_termination_values)
            .pipe(filter_extreme_rating_diffs)
            .pipe(drop_columns_from_dataframe, COLUMNS_TO_DROP_GAMES) # It's important to drop columns at the end of the cleaning process, because some of them are used in the filtering process
            ) 


def drop_columns_from_dataframe(df : pd.DataFrame, columns_to_drop):
    return df.drop(columns=columns_to_drop)

def filter_by_equal_conditional_expression(df : pd.DataFrame, column, value):
    return df[df[column] == value]

def filter_games_with_same_white_and_black_player(df : pd.DataFrame):
    return df[df['white'] != df['black']]

def filter_extreme_num_plies_values(df : pd.DataFrame):
    return df[(df['num_plies'] < SUPERIOR_LIMIT_NUM_PLIES ) & (df['num_plies'] > INFERIOR_LIMIT_NUM_PLIES)]

def filter_only_interesting_termination_values(df : pd.DataFrame):
    return df[(df['termination'] != 'Abandoned') & (df['termination'] != 'Rules infraction')]

def filter_extreme_rating_diffs(df : pd.DataFrame):
    return df[(df['white_rating_diff'] < SUPERIOR_LIMIT_RATING_DIFF) & (df['black_rating_diff'] < -SUPERIOR_LIMIT_RATING_DIFF)]