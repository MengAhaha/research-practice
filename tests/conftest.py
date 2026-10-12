import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    """小型合成数据：target 主要由 x0（正）和 x1（负）决定，x2、x3 是噪声。"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["x0", "x1", "x2", "x3"])
    X["target"] = 3 * X["x0"] - 2 * X["x1"] + rng.normal(scale=0.5, size=len(X))
    return X
