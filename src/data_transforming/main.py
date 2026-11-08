import pandas as pd
import numpy as np

DECIDED_CP = 800
DECIDED_MAX_MATE_TO_SEE = 10

def build_features_dataframe(moves : pd.DataFrame, games:pd.DataFrame) -> pd.DataFrame:
    moves_features = build_move_features(moves)
    games_features = build_game_features(games)
    return games_features.merge(moves_features, on=['game_id', 'is_white'])

def build_move_features(moves : pd.DataFrame):
    m = moves.sort_values(['game_id', 'ply'])
    return (
        m.pipe(add_who_moved)
            .pipe(fill_checkmating_moves)
            .pipe(add_win_percentages)
            .pipe(add_win_loss)
            .pipe(add_unseen_mate)
            .pipe(sumarize_per_player)
    )
    
def add_who_moved(moves : pd.DataFrame):
    moves['is_white'] = (moves['ply'] % 2 == 1).astype(int)
    return moves

def fill_checkmating_moves(m: pd.DataFrame) -> pd.DataFrame:
    previous_mate = m.groupby('game_id')['mate'].shift(1)
    is_checkmating_move = m['eval'].isna() & m['mate'].isna()

    m.loc[is_checkmating_move, 'mate'] = previous_mate[is_checkmating_move]
    return m

def add_win_percentages(moves : pd.DataFrame):
    cliped = moves['eval'].clip(-DECIDED_CP, DECIDED_CP)
    mate_clipped = np.where(moves['mate'] > 0, DECIDED_CP, -DECIDED_CP)
    cliped = cliped.where(cliped.notna(), mate_clipped)

    moves['win%_white_after'] = win_pct(cliped)
    moves['win%_white_before'] = moves.groupby(['game_id'])['win%_white_after'].shift(1)
    moves['win%_white_before'] = moves['win%_white_before'].fillna(win_pct(20))
    return moves

def add_win_loss(moves : pd.DataFrame):
    win_before = moves['win%_white_before']
    win_after = moves['win%_white_after']

    moves['win_loss'] = np.where(moves['is_white'], win_before-win_after, win_after-win_before)
    moves['win_loss'] = moves['win_loss'].clip(0)
    return moves

def add_unseen_mate(moves : pd.DataFrame):
    previous_mate = moves.groupby('game_id')['mate'].shift(1)

    had_mate = np.where(moves['is_white'], previous_mate>0, previous_mate<0)
    mate_was_short = previous_mate.abs() <= DECIDED_MAX_MATE_TO_SEE
    still_has_mate = np.where(moves['is_white'], moves['mate']>0, moves['mate']<0)
    
    moves['unseen_mate'] = (had_mate & ~still_has_mate & mate_was_short).astype(int)
    return moves

def sumarize_per_player(moves : pd.DataFrame):
    return (
        moves.groupby(['game_id', 'is_white'])
         .agg(**{
             'avg_win%_loss':         ('win_loss', 'mean'),
             'highest_win%_loss':     ('win_loss', 'max'),
             'num_unseen_checkmates': ('unseen_mate', 'sum'),
         })
         .reset_index()
    )

def build_game_features(games : pd.DataFrame):
    g = games.copy()
    g['termination_normal'] = g['termination'] == 'Normal'
    g['termination_time'] = g['termination'] == 'Time forfeit'

    times = g['time_control'].str.split('+', expand=True).astype(int)
    g['base_time'] = times[0].astype(int)
    g['increment_per_move'] = times[1].astype(int)

    common_columns = ['game_id', 'num_plies', 'base_time', 'increment_per_move',
                  'termination_normal', 'termination_time']

    white = g[common_columns].copy()
    white['is_white'] = 1
    white['elo'] = g['white_elo']
    white['result_win'] = (g['result'] == '1-0').astype(int)
    white['result_loss'] = (g['result'] == '0-1').astype(int)
    white['result_draw'] = 1-white['result_win']-white['result_loss']

    black = g[common_columns].copy()
    black['is_white'] = 0
    black['elo'] = g['black_elo']
    black['result_win'] = (g['result'] == '0-1').astype(int)
    black['result_loss'] = (g['result'] == '1-0').astype(int)
    black['result_draw'] = 1-black['result_win']-black['result_loss']

    return pd.concat([white, black], ignore_index=True)

def win_pct(cp):
    return 50 + 50 * (2 / (1 + np.exp(-0.00368208 * cp)) - 1)
