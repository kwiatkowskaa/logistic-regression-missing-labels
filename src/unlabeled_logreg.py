import numpy as np
from src.fista_logreg import FISTALogisticLasso

class UnlabeledLogReg:
    """
    Logistic regression with missing labels using FISTA.

    Parameters
    ----------
    method (str): Method for completing Y:
        - "self_training"
        - "soft_labels"
        - "naive_method"
        Default = "self_training"
    max_iter (int): Maximum number of iterations in methods 'self_training'' and 'soft_labels'. Default = 10
    tol (float): Convergence tolerance. Default = 1e-4
    threshold (float): Threshold used to convert predicted probabilities into binary labels. Default=0.5
    fista_params (dict, optional): Parameters passed to FISTALogisticLasso.
    """

    def __init__(self, method="self_training", max_iter=10, tol=1e-4, threshold=0.5, fista_params=None):

        if method not in ["self_training", "soft_labels", "naive_method"]:
            raise ValueError("method must be 'self_training', 'soft_labels' or 'naive_method' ")

        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.fista_params = fista_params if fista_params is not None else {}

        self.model_ = None
        self.beta_ = None
        self.Y_completed_ = None

    def fit(self, X, Y_obs):
        """
        Fit model with missing labels.
        """

        X = np.asarray(X)
        Y_obs = np.asarray(Y_obs)

        labeled_mask = Y_obs != -1
        unlabeled_mask = Y_obs == -1

        X_l = X[labeled_mask]
        Y_l = Y_obs[labeled_mask]
        X_u = X[unlabeled_mask]


        if self.method == "self_training":
            X_completed, Y_completed = self._self_training(X_l, Y_l, X_u, unlabeled_mask, X, Y_obs)

        elif self.method == "soft_labels":
            X_completed, Y_completed = self._soft_labels(X_u, unlabeled_mask, X, Y_obs)

        elif self.method == 'naive_method':
            X_completed, Y_completed = self._naive_method(X_l, Y_l)

        model = FISTALogisticLasso(**self.fista_params)
        model.fit(X_completed, Y_completed)

        self.model_ = model
        self.beta_ = model.beta_
        self.Y_completed_ = Y_completed

        return self

    def _self_training(self, X_l, Y_l, X_u, unlabeled_mask, X, Y_obs):
        """
        Perform self-training to iteratively complete missing labels.

        At each iteration:
        - Train model on currently labeled data
        - Predict labels for unlabeled samples
        - Add predicted labels to the training set
        """

        Y_completed = Y_obs.copy()
        X_completed = X.copy()

        for _ in range(self.max_iter):
            
            model = FISTALogisticLasso(**self.fista_params)
            model.fit(X_l, Y_l)

            proba = model.predict_proba(X_u)
            Y_u_pred = (proba >= self.threshold).astype(int)

            Y_completed[unlabeled_mask] = Y_u_pred

            X_l = X
            Y_l = Y_completed

        return X_completed, Y_completed

    def _soft_labels(self, X_u, unlabeled_mask, X, Y_obs):
        """
        Perform soft-label training.

        Missing labels are initialized with 0.5 and iteratively updated
        using predicted probabilities until convergence or max_iter is reached.
        """

        Y_completed = Y_obs.copy().astype(float)

        Y_completed[unlabeled_mask] = 0.5

        X_completed = X.copy()

        for _ in range(self.max_iter):

            Y_old = Y_completed.copy()

            model = FISTALogisticLasso(**self.fista_params)
            model.fit(X, Y_completed)

            proba = model.predict_proba(X_u)

            Y_completed[unlabeled_mask] = proba

            if np.linalg.norm(Y_completed - Y_old) < self.tol:
                break

        Y_completed = (Y_completed >= self.threshold).astype(int)

        return X_completed, Y_completed
    
    def _naive_method(self, X_l, Y_l):
        """
        Train only on labeled data, ignoring unlabeled samples.
        """

        X_completed = X_l.copy()
        Y_completed = Y_l.copy()

        return X_completed, Y_completed

    def predict(self, X):
        """
        Predict binary labels for new data.
        """

        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """

        return self.model_.predict_proba(X)