import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, Matern
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline

SEASON_MAP = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}

def load_data():
    """
    Loads and preprocesses training and test data.
    Key improvements:
      - KNN imputation (respects cross-country correlations)
      - Ordinal season encoding + cyclical sin/cos features
      - Mean-of-neighbors and cross-country ratio features
    """
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")

    print("Training data shape:", train_df.shape)
    print("Test data shape:    ", test_df.shape)

    # ------------------------------------------------------------------ #
    # 1. Encode season: ordinal + cyclical                                 #
    # ------------------------------------------------------------------ #
    for df in [train_df, test_df]:
        df['season_ord'] = df['season'].map(SEASON_MAP)
        df['season_sin'] = np.sin(2 * np.pi * df['season_ord'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season_ord'] / 4)
        df.drop('season', axis=1, inplace=True)

    # ------------------------------------------------------------------ #
    # 2. Separate target BEFORE imputation to avoid leakage               #
    # ------------------------------------------------------------------ #
    # Keep only rows with known price_CHF for training
    train_known = train_df.dropna(subset=['price_CHF']).copy()
    y_train = train_known['price_CHF'].values

    # Feature columns = everything except price_CHF
    feature_cols = [c for c in train_df.columns if c != 'price_CHF']
    X_train_raw = train_known[feature_cols].copy()
    X_test_raw  = test_df[feature_cols].copy()

    # ------------------------------------------------------------------ #
    # 3. KNN imputation (fit on train features only → transform both)     #
    # ------------------------------------------------------------------ #
    # KNN imputer uses the k nearest complete rows to fill gaps,
    # which is much better than mean for correlated price data
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    X_train_imp = pd.DataFrame(X_train_imp, columns=feature_cols)
    X_test_imp  = pd.DataFrame(X_test_imp,  columns=feature_cols)

    # ------------------------------------------------------------------ #
    # 4. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    price_cols = [c for c in feature_cols if c.startswith('price_')]

    for df in [X_train_imp, X_test_imp]:
        # Row mean/std of all other countries' prices — strong signal
        df['price_mean']   = df[price_cols].mean(axis=1)
        df['price_std']    = df[price_cols].std(axis=1)
        df['price_median'] = df[price_cols].median(axis=1)
        df['price_min']    = df[price_cols].min(axis=1)
        df['price_max']    = df[price_cols].max(axis=1)
        df['price_range']  = df['price_max'] - df['price_min']

        # Pairwise interactions with most correlated neighbors
        # Germany and France are Switzerland's direct electricity grid neighbors
        for neighbor in ['price_GER', 'price_FRA']:
            if neighbor in df.columns:
                df[f'{neighbor}_x_mean'] = df[neighbor] * df['price_mean']
                df[f'{neighbor}_x_season_sin'] = df[neighbor] * df['season_sin']

    X_train = X_train_imp.values
    X_test  = X_test_imp.values

    print(f"\nFinal shapes — X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0]  == 100

    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.model  = None

    def _build_stacking_model(self):
        """
        Stacking ensemble:
        - Base learners capture different aspects of the data
        - RidgeCV as final estimator is regularized and stable
        """
        estimators = [
            ('ridge',  RidgeCV(alphas=np.logspace(-3, 3, 20))),
            ('gbr',    GradientBoostingRegressor(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.8,
                            min_samples_leaf=3,
                            random_state=42)),
            ('rf',     RandomForestRegressor(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1)),
        ]
        # RidgeCV final estimator: linear blend of base learners, won't overfit
        final = RidgeCV(alphas=np.logspace(-3, 3, 20))
        return StackingRegressor(
            estimators=estimators,
            final_estimator=final,
            cv=5,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)

        # Cross-validate each candidate to pick the best single model
        # as a sanity check before committing to stacking
        candidates = {
            'RidgeCV':  Pipeline([('m', RidgeCV(alphas=np.logspace(-3, 3, 20)))]),
            'GBR':      Pipeline([('m', GradientBoostingRegressor(
                                        n_estimators=300, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42))]),
            'Stacking': Pipeline([('m', self._build_stacking_model())]),
        }

        print("\nCross-validation R2 scores (5-fold):")
        best_score = -np.inf
        best_name  = None
        for name, pipe in candidates.items():
            scores = cross_val_score(pipe, X_scaled, y_train,
                                     cv=5, scoring='r2', n_jobs=-1)
            mean_r2 = scores.mean()
            print(f"  {name:12s}: {mean_r2:.4f} ± {scores.std():.4f}")
            if mean_r2 > best_score:
                best_score = mean_r2
                best_name  = name

        print(f"\nSelecting: {best_name} (CV R2 = {best_score:.4f})")
        self.model = candidates[best_name]
        self.model.fit(X_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        y_pred   = self.model.predict(X_scaled)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()

    model = Model()
    model.fit(X_train=X_train, y_train=y_train)

    y_pred = model.predict(X_test)

    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")