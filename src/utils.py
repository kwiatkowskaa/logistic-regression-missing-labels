import numpy as np
import pandas as pd

from scipy.io import arff
from ucimlrepo import fetch_ucirepo 

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_PATH / "data"


def load_dataset(name):
    """
    Load a dataset by name. Supported datasets are "biodeg", "higgs", "pendigits" and "magic".
    
    Parameters:
    -----------
    name (str): The name of dataset to load.

    Returns:
    --------
    X (pd.DataFrame): DataFrame conatining the features of the dataset.
    y (pd.Series): Series containing the target variable.
    """

    if name == "biodeg":
        data, _ = arff.loadarff(DATA_PATH / "biodeg.arff")
        df = pd.DataFrame(data)
        y = df["Class"]
        y = y.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        y = y.map({"1":1, "2":0})
        X = df.drop(columns=["Class"])

    elif name == "higgs":
        data, _ = arff.loadarff(DATA_PATH / "higgs.arff")
        df = pd.DataFrame(data)
        y = df["class"]
        y = y.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        y = y.astype(int)
        X = df.drop(columns=["class"])

    elif name == "pendigits":
        data, _ = arff.loadarff(DATA_PATH / "pendigits.arff")
        df = pd.DataFrame(data)
        y = df["binaryClass"]
        y = y.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        y = y.map({"N":0, "P":1})
        X = df.drop(columns=["binaryClass"])

    elif name == "magic":
        dataset = fetch_ucirepo(id=159)
        X = dataset.data.features
        y = dataset.data.targets
        y = y.iloc[:, 0]
        y = y.map({"g":1, "h":0})
    
    elif name == "spambase":
        dataset = fetch_ucirepo(id=94)
        X = dataset.data.features
        y = dataset.data.targets
        y = y.iloc[:, 0]
        y = y.astype(int)

    elif name == "htru2":
        dataset = fetch_ucirepo(id=372)
        X = dataset.data.features
        y = dataset.data.targets.iloc[:, 0]
        y = y.astype(int)

    return X, y


def prepare_dataset(X, threshold=0.9):
    """
    Remove rows with missing values and remove features that have a correlation higher
    than the specified treshold.
    
    Parameters:
    -----------
    X (pd.DataFrame): Input DataFrame containing the features.
    threshold (float): Correlation threshold above which features will be removed.

    Returns:
    --------
    X_filtered (pd.DataFrame): A DataFrame without missing values and correlated features removed.
    to_drop (list): List of features that were removed due to high correlation.
    """

    X = X.dropna()

    corr = X.corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_filtered = X.drop(columns=to_drop)

    return X_filtered, to_drop



def dataset_summary(X, y, name):
    """
    Print summary of datasets
    """

    print("-"*40)
    print(f"Dataset: {name}")
    print("-"*40)

    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\n")
