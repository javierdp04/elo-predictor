import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from pathlib import Path
from tqdm_joblib import tqdm_joblib

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

COLS_TO_DROP = ['game_id']
SAMPLE_FRAC = 0.5

def build_main_model(n_estimators=100, max_depth=5, min_samples_leaf=1, max_features=0.75):
    main_pipeline = Pipeline([
        ('drop', ColumnTransformer([
            ('drop', 'drop', COLS_TO_DROP)
        ], remainder='passthrough')),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, max_features=max_features
                                        ))
    ])
    return main_pipeline

if __name__ == '__main__':
    trainData = pd.read_csv(DATA_DIR / 'features.csv')

    X = trainData.drop(columns=['elo'])
    y = trainData['elo']
    groups = X['game_id']

    splitter = GroupShuffleSplit(n_splits=1, train_size=SAMPLE_FRAC, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    main_pipeline = build_main_model()
 
    param_distributions = {
    "regressor__n_estimators": [300],
    "regressor__max_depth": [5, 10, 15, 20, 25],
    "regressor__min_samples_leaf": [5, 7, 10, 13, 15],
    "regressor__max_features": [0.5, 0.6, 0.7, 0.8],
    "regressor__max_samples": [0.3, 0.5, 0.6, 0.7, 0.8],
    }

    cv = 5
    n_iter = 80        
    search = RandomizedSearchCV(main_pipeline, param_distributions, n_iter=n_iter,
                                cv=GroupKFold(n_splits=cv), scoring='neg_mean_absolute_error',
                                n_jobs=-1, random_state=0)
    with tqdm_joblib(total=n_iter * cv, desc="Ajustando hiperparametros"):
        search.fit(X_train, y_train, groups=X_train['game_id'])

    print("Mejores parámetros:", search.best_params_)
    print("Mejor MAE CV:", -search.best_score_)

    y_predict = search.predict(X_test)
    print("MAE en test:", mean_absolute_error(y_test, y_predict))
