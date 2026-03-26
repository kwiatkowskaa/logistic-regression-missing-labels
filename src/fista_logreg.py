import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score

class FISTALogisticLasso:
    """
    Initialize the FISTA-based logistic Lasso model.

    Parameters
    ----------
    lambda_values (array-like, optional): Sequence of regularization strengths to evaluate during validation.
    lam (float): Regularization strength used when auto_validate=False. Default=1.0
    max_iter (int): Maximum number of FISTA iterations. Default=100
    tol (float): Convergence tolerance for stopping criterion. Default=1e-6
    """
    
    def __init__(self, lambda_values=None, lam=1.0, max_iter=100, tol=1e-6):

        if lambda_values is None:
            self.lambda_values = np.logspace(-3, 0, 20)
        else:
            self.lambda_values = lambda_values

        self.lam = lam

        self.max_iter = max_iter
        self.tol = tol

        self.measure_ = None

        self.beta_ = None
        self.best_lambda_ = None

        self.betas_ = []
        self.scores_ = []
    

    def fit(self, X_train, y_train, auto_validate=False, X_valid=None, y_valid=None, measure="f1"):
        """
        Fit the model using FISTA and optionally select the best lambda via validation.
        """

        self.betas_ = []
        self.scores_ = []
        self.measure_ = measure

        if not auto_validate:
            self.beta_ = self._fista_optimize(X_train, y_train, self.lam)
            return self

        if X_valid is None or y_valid is None:
            raise ValueError("X_valid and y_valid are required with auto_validate=True")

        for lam in self.lambda_values:

            beta = self._fista_optimize(X_train, y_train, lam)
            self.beta_ = beta

            score = self.validate(X_valid, y_valid, measure=measure)

            self.scores_.append(score)
            self.betas_.append(beta)

        best_idx = np.argmax(self.scores_)

        self.best_lambda_ = self.lambda_values[best_idx]
        self.beta_ = self.betas_[best_idx]

    
    def validate(self, X_valid, y_valid, measure='f1'):
        """
        Evaluate the model on validation data using a chosen metric.
        """

        proba = 1 / (1 + np.exp(-(X_valid @ self.beta_)))
        y_pred = (proba >= 0.5).astype(int)

        if measure == "precision":
            return precision_score(y_valid, y_pred)

        elif measure == "recall":
            return recall_score(y_valid, y_pred)

        elif measure == "f1":
            return f1_score(y_valid, y_pred)

        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_valid, y_pred)

        elif measure == "roc_auc":
            return roc_auc_score(y_valid, proba)
        
        elif measure == "pr_auc":
            return average_precision_score(y_valid, proba)
        
        else:
            raise ValueError(f"Unknown measure: {measure}")


    def _fista_optimize(self, X_train, y_train, lambda_value):
        """
        Optimize logistic loss with L1 regularization using FISTA.
        """

        beta = np.zeros(X_train.shape[1])
        y = beta.copy()

        t = 1
        L = np.linalg.norm(X_train, 2)**2 / 4
        
        for i in range(self.max_iter):

            beta_old = beta.copy()
            gradient = self._gradient(X_train, y_train, y)

            beta = self._soft_thresholding(y - gradient / L, lambda_value / L)
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = beta + ((t - 1) / t_new) * (beta - beta_old)
            t = t_new

            if np.linalg.norm(beta - beta_old) < self.tol:
                break

        return beta
        
        
    
    def predict_proba(self, X_test):
        """
        Compute predicted probabilities for input samples.
        """
        if self.beta_ is None:
            raise ValueError("Model not trained. Run fit() first.")
        
        z = X_test @ self.beta_
        return 1 / (1 + np.exp(-z))
    
    
    def predict(self, X_test, threshold=0.5):
        """
        Predict binary class labels for input samples.
        """
        return (self.predict_proba(X_test) >= threshold).astype(int)


    def plot(self):
        """
        Plot validation scores as a function of lambda values.
        """

        if self.beta_ is None:
            raise ValueError("Model not trained. Run fit() first.")
        if self.best_lambda_ is None:
            raise ValueError("Model was trained without lambda validation. Run fit(auto_validate=True) first.")

        plt.figure(figsize=(7, 4))
        plt.plot(
            self.lambda_values,
            self.scores_,
            marker="o",
            linewidth=2,
            color="#785EF0",
            label="validation score"
        )

        best_idx = np.argmax(self.scores_)
        best_lambda = self.lambda_values[best_idx]

        plt.axvline(
            best_lambda,
            linestyle="--",
            linewidth=1,
            color="black",
            label=f"best λ = {best_lambda:.2e}"
        )

        plt.xscale("log")
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(self.measure_, fontsize=12)
        plt.title("Validation score vs regularization", fontsize=13)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()


    def plot_coefficients(self):
        """
        Plot coefficient paths as a function of lambda values.
        """

        if self.beta_ is None:
            raise ValueError("Model not trained. Run fit() first.")
        if self.best_lambda_ is None:
            raise ValueError("Model was trained without lambda validation. Run fit(auto_validate=True) first.")

        betas = np.array(self.betas_)

        plt.figure(figsize=(7, 4))

        for j in range(betas.shape[1]):
            plt.plot(
                self.lambda_values,
                betas[:, j],
                linewidth=1
            )

        plt.xscale("log")
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel("Coefficient value", fontsize=12)
        plt.title("Regularization path", fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()


    def _gradient(self, X, y, beta):
        """
        Compute the gradient of the logistic loss.
        """
        Xb = X @ beta
        z = 1 / (1 + np.exp(-Xb))
        return X.T @ (z - y)
    

    def _soft_thresholding(self, v, alpha):
        """
        Apply the soft-thresholding operator for L1 regularization.
        """
        return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)
