import numpy as np
import pandas as pd
from scipy.stats import rankdata

def generate_mcar(X, Y, c):
    """
    Generate Missing Completely At Random (MCAR) labels.

    This function simulates missingness in the target variable `Y` by randomly
    masking a fraction of its values. Each label has probability `c` of being
    replaced with -1, indicating a missing label.
    """
    S = np.random.binomial(n=1, p=c, size=X.shape[0])
    Y_obs = Y.copy()
    Y_obs[S == 1] = -1
    return X, Y_obs


def generate_mar1(X, Y, c, j=1):
    """
    Generate Missing At Random (MAR) labels based on one feature.

    This function simulates missingness in the target variable `Y` such that
    the probability of a label being missing depends on the values of a
    specified feature column `X[:, j]`. Higher deviations from the mean of the
    feature correspond to lower probabilities of missingness.

    The missingness probability is computed as:
        p_i = exp(- (x_i - mu)^2 / (2 * sigma^2))
    and then scaled to achieve the desired overall missingness fraction `c`.
    """
    x = X.iloc[:, j]
    mu = np.mean(x)
    sigma = np.std(x)
    p = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    p_scaled = p * (c / p.mean())
    p_scaled - np.clip(p_scaled, a_min=0, a_max=1)
    S = np.random.binomial(1, p_scaled)
    Y_obs = Y.copy()
    Y_obs[S == 1] = -1
    return X, Y_obs


def generate_mar2(X, Y, c):
    """
    Generate Missing At Random (MAR) labels using a linear combination of features.

    This function simulates missingness in the target variable `Y` such that
    the probability of a label being missing depends on a logistic function
    of a random linear combination of standardized features. The probabilities
    are scaled to match the desired overall missing fraction `c`.
    """
    X_std = (X - X.mean()) / X.std()
    beta = np.random.randn(X_std.shape[1]) * 0.5

    linear_pred = X_std.values.dot(beta)
    p = 1.0 / (1.0 + np.exp(-linear_pred))
    p = p * (c / p.mean())
    p = np.clip(p, 0, 1)

    S = np.random.binomial(1, p)
    Y_obs = Y.copy()
    Y_obs[S == 1] = -1

    return X, Y_obs


def generate_mnar(X, Y, c):
    """
    Generate Missing Not At Random (MNAR) labels based on target values.

    This function simulates missingness in the target variable `Y` such that
    the probability of a label being missing depends on the rank of the label
    itself. Higher-ranked labels have a higher probability of being missing,
    creating a Missing Not At Random (MNAR) pattern.
    """
    ranks = rankdata(Y.values)
    p_raw = ranks / ranks.sum()
    p = p_raw * (c * len(Y))
    p = np.clip(p, 0, 1)
    print(p)
    S = np.random.binomial(1, p)
    print(S)
    Y_obs = Y.copy()
    Y_obs[S == 1] = -1
    return X, Y_obs
