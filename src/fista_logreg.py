import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score

class FISTALogisticLasso:
    """
    Implementation of Logistic Lasso Regression using FISTA.
    """
    
    def __init__(self, lambda_values=None, max_iter=1000, tol=1e-6):

        if lambda_values is None:
            self.lambda_values = np.logspace(-4, 0, 10)
        else:
            self.lambda_values = lambda_values

        self.max_iter = max_iter
        self.tol = tol

        self.beta_ = None
        self.best_lambda_ = None

        self.betas_ = []
        self.scores_ = []
    

    def fit(self, X_train, y_train):

        if self.best_lambda_ is None:
            raise ValueError("You must call validate() to find the best lambda before fitting the model.")

        beta = self.fista_optimize(X_train, y_train, self.best_lambda_)
        self.beta_ = beta


    def fista_optimize(self, X_train, y_train, lambda_value):

        beta = np.zeros(X_train.shape[1])
        y = beta.copy()

        t = 1
        L = np.linalg.norm(X_train, 2)**2 / 4
        
        for _ in range(self.max_iter):

            beta_old = beta.copy()
            gradient = self._gradient(X_train, y_train, y)
            beta = self._soft_thresholding(y - gradient / L, lambda_value / L)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = beta + ((t - 1) / t_new) * (beta - beta_old)
            t = t_new

            if np.linalg.norm(beta - beta_old) < self.tol:
                break

        return beta

    
    def validate(self, X_valid, y_valid, measure='precision'):

        self.betas_ = []
        self.scores_ = []

        for lam in self.lambda_values:

            beta = self.fista_optimize(X_valid, y_valid, lam)

            proba = 1 / (1 + np.exp(-(X_valid @ beta)))
            y_pred = (proba >= 0.5).astype(int)
            score = self.score(y_valid, y_pred, proba, measure=measure)

            self.scores_.append(score)
            self.betas_.append(beta)

        self.best_lambda_ = self.lambda_values[np.argmax(self.scores_)]
    

    def score(self, y_valid, y_pred, proba, measure="precision"):
        if measure == "precision":
            return precision_score(y_valid, y_pred)

        if measure == "recall":
            return recall_score(y_valid, y_pred)

        if measure == "f1":
            return f1_score(y_valid, y_pred)

        if measure == "balanced_accuracy":
            return balanced_accuracy_score(y_valid, y_pred)

        if measure == "roc_auc":
            return roc_auc_score(y_valid, proba)
        
    
    def predict_proba(self, X_test):
        if self.beta_ is None:
            raise ValueError("Model not trained. Run validate() and fit() first.")
        
        z = X_test @ self.beta_
        return 1 / (1 + np.exp(-z))
    
    
    def predict(self, X_test, threshold=0.5):
        return (self.predict_proba(X_test) >= threshold).astype(int)


    def plot(self, measure):

        plt.figure(figsize=(6,4))
        plt.plot(
            self.lambda_values,
            self.scores_,
            marker="o",
            linewidth=2
        )

        best_idx = np.argmax(self.scores_)
        best_lambda = self.lambda_values[best_idx]

        plt.axvline(
            best_lambda,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"best λ = {best_lambda:.2e}"
        )

        plt.xscale("log")
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(measure, fontsize=12)
        plt.title("Validation score vs regularization", fontsize=13)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()


    def plot_coefficients(self):

        betas = np.array(self.betas_)

        plt.figure(figsize=(6,4))

        for j in range(betas.shape[1]):

            plt.plot(
                self.lambda_values,
                betas[:, j],
                linewidth=1
            )

        plt.xscale("log")
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel("Coefficient value", fontsize=12)
        plt.title("Coefficients path", fontsize=13)

        plt.grid(alpha=0.3)

        plt.tight_layout()


    def _gradient(self, X, y, beta):
        z = 1 / (1 + np.exp(-X @ beta))
        return X.T @ (z - y)
    

    def _soft_thresholding(self, v, alpha):
        return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)
