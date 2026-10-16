import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD


def build_ratings(n_users: int = 30, n_items: int = 20, n_ratings: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "user": rng.integers(0, n_users, n_ratings),
            "item": rng.integers(0, n_items, n_ratings),
            "rating": rng.integers(1, 6, n_ratings),
        }
    )
    return df.drop_duplicates(["user", "item"]).reset_index(drop=True)


def train_svd(ratings: pd.DataFrame) -> SVD:
    data = Dataset.load_from_df(ratings[["user", "item", "rating"]], Reader(rating_scale=(1, 5)))
    model = SVD(random_state=0)
    model.fit(data.build_full_trainset())
    return model
