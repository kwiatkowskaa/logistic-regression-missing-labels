import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from src.fista_logreg import FISTALogisticLasso

class UnlabeledLogReg:

    def __init__(self, method="pseudo_labeling", n_clusters=10, fista_params=None):
        """
        Semi-supervised logistic regression model handling missing labels.

        This class implements different strategies for dealing with unlabeled
        samples (denoted by -1 in Y_obs), and then trains a logistic regression
        model using the completed dataset.

        Available methods include:
        - pseudo-labeling
        - k-means majority voting
        - naive (ignore unlabeled data)

        Parameters
        ----------
        method : str, optional
            Strategy used to handle unlabeled data. Must be one of:
            - "pseudo_labeling": Train a model on labeled data and predict missing labels
            - "kmeans_majority": Assign labels based on cluster majority voting
            - "naive": Use only labeled data
            Default = "pseudo_labeling"

        n_clusters : int, optional
            Number of clusters used in k-means (only relevant for "kmeans_majority").
            Default = 10

        fista_params : dict, optional
            Parameters passed to the FISTALogisticLasso model.
            If None, default parameters are used.

        Attributes
        ----------
        model_ : FISTALogisticLasso
            Trained logistic regression model.
        """

        if method not in ["pseudo_labeling", "kmeans_majority", "naive"]:
            raise ValueError("method must be 'pseudo_labeling', 'kmeans_majority' or 'naive'")

        self.method = method
        self.n_clusters = n_clusters
        self.fista_params = fista_params if fista_params is not None else {}

        self.model_ = None

    def fit(self, X, Y_obs, X_val, y_val):
        """
        Fit the model using partially labeled data.

        This method separates labeled and unlabeled samples, applies the selected
        strategy to complete missing labels, and trains a logistic regression model.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Y_obs : array-like
            Observed labels. Unlabeled samples should be marked as -1.

        X_val : array-like
            Validation feature matrix used for model selection.

        y_val : array-like
            Validation labels.

        Returns
        -------
        self : object
            Fitted model instance.
        """

        X = np.asarray(X)
        Y_obs = np.asarray(Y_obs)

        labeled_mask = Y_obs != -1
        unlabeled_mask = Y_obs == -1

        X_l = X[labeled_mask]

        if X_l.shape[0] < 5:
            print(f"Too few labeled samples ({X_l.shape[0]}), skipping...")
            return self
        
        Y_l = Y_obs[labeled_mask]

        X_u = X[unlabeled_mask]

        if len(X_l) < 5:
            raise ValueError("Too few labeled samples")

        if self.method == "pseudo_labeling":
            X_completed, Y_completed = self._pseudo_labeling(X, Y_obs, X_l, Y_l, X_u, unlabeled_mask)

        elif self.method == "kmeans_majority":
            X_completed, Y_completed = self._kmeans_majority(X, Y_obs, labeled_mask, unlabeled_mask)

        elif self.method == "naive":
            X_completed, Y_completed = X_l, Y_l

        model = FISTALogisticLasso(**self.fista_params)
        model.fit(X_completed, Y_completed, auto_validate=True, X_valid=X_val, y_valid= y_val)
        self.model_ = model

        return self

    def _pseudo_labeling(self, X, Y_obs, X_l, Y_l, X_u, unlabeled_mask):
        """
        Fill missing labels using pseudo-labeling.

        A logistic regression model is trained on labeled data and used
        to predict labels for unlabeled samples.

        Parameters
        ----------
        X : array-like
            Full dataset.

        Y_obs : array-like
            Observed labels with missing values.

        X_l : array-like
            Labeled feature subset.

        Y_l : array-like
            Labels for labeled data.

        X_u : array-like
            Unlabeled feature subset.

        unlabeled_mask : array-like (bool)
            Mask indicating unlabeled samples.

        Returns
        -------
        tuple
            (X, Y_completed) where missing labels are filled.
        """

        Y_completed = Y_obs.copy()

        model = LogisticRegression(max_iter=1000)
        model.fit(X_l, Y_l)
        
        if X_u.shape[0] > 0:
            Y_completed[unlabeled_mask] = model.predict(X_u)

        return X, Y_completed

    def _kmeans_majority(self, X, Y_obs, labeled_mask, unlabeled_mask):
        """
        Fill missing labels using k-means clustering and majority voting.

        Data is clustered using k-means, and unlabeled samples receive
        the majority label of labeled samples within the same cluster.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Y_obs : array-like
            Observed labels.

        labeled_mask : array-like (bool)
            Mask indicating labeled samples.

        unlabeled_mask : array-like (bool)
            Mask indicating unlabeled samples.

        Returns
        -------
        tuple
            (X, Y_completed) with filled labels.
        """

        Y_completed = Y_obs.copy()

        all_X = X
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(all_X)

        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            labeled_in_cluster = cluster_mask & labeled_mask
            unlabeled_in_cluster = cluster_mask & unlabeled_mask

            if np.sum(labeled_in_cluster) == 0:
                continue

            majority_label = np.round(np.mean(Y_obs[labeled_in_cluster]))
            if np.sum(unlabeled_in_cluster) > 0:
                Y_completed[unlabeled_in_cluster] = majority_label

        return X, Y_completed

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            Predicted class labels.
        """

        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        array-like
            Predicted probabilities.
        """

        return self.model_.predict_proba(X)
