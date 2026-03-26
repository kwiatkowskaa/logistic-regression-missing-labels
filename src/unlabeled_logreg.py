import numpy as np
from sklearn.cluster import KMeans
from src.fista_logreg import FISTALogisticLasso
from sklearn.linear_model import LogisticRegression

class UnlabeledLogReg:

    def __init__(self, method="self_training", n_clusters=10, fista_params=None):

        if method not in ["pseudo_labeling", "kmeans_majority", "naive"]:
            raise ValueError("method must be 'pseudo_labeling', 'kmeans_majority' or 'naive'")

        self.method = method
        self.n_clusters = n_clusters
        self.fista_params = fista_params if fista_params is not None else {}

        self.model_ = None

    def fit(self, X, Y_obs, X_val, y_val):

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

        Y_completed = Y_obs.copy()

        model = LogisticRegression(max_iter=1000) ## można uzyc FISTALogisticLasso(**self.fista_params) ale to daje podobne wyniki
        model.fit(X_l, Y_l)
        
        if X_u.shape[0] > 0:
            Y_completed[unlabeled_mask] = model.predict(X_u)

        return X, Y_completed

    def _kmeans_majority(self, X, Y_obs, labeled_mask, unlabeled_mask):

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
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
