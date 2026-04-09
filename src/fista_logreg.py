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
    fit_intercept (bool): Whether to fit an intercept term. Default=False
    """
    
    def __init__(self, lambda_values=None, lam=1.0, max_iter=100, tol=1e-6, fit_intercept=False):

        if lambda_values is None:
            self.lambda_values = np.logspace(-3, 1, 20)
        else:
            self.lambda_values = lambda_values

        self.lam = lam

        self.max_iter = max_iter
        self.tol = tol

        self.fit_intercept = fit_intercept
        self.intercept_ = 0.0

        self.measure_ = None

        self.beta_ = None
        self.best_lambda_ = None

        self.betas_ = []
        self.scores_ = []
        self.intercepts_ = []
    

    def fit(self, X_train, y_train, auto_validate=False, X_valid=None, y_valid=None, measure="f1"):
        """
        Train the model using FISTA, optionally selecting the best lambda.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        auto_validate : bool
            Whether to perform validation-based lambda selection.
        X_valid : array-like
            Validation features.
        y_valid : array-like
            Validation labels.
        measure : str
            Metric used for validation.
        """

        self.betas_ = []
        self.intercepts_ = []
        self.scores_ = []
        self.measure_ = measure

        if not auto_validate:
            beta, intercept = self._fista_optimize(X_train, y_train, self.lam)
            self.beta_ = beta
            self.intercept_ = intercept
            return self

        if X_valid is None or y_valid is None:
            raise ValueError("X_valid and y_valid are required with auto_validate=True")

        for lam in self.lambda_values:

            beta, intercept = self._fista_optimize(X_train, y_train, lam)

            self.beta_ = beta
            self.intercept_ = intercept

            score = self.validate(X_valid, y_valid, measure=measure)

            self.scores_.append(score)
            self.betas_.append(beta)
            self.intercepts_.append(intercept)

        best_idx = np.argmax(self.scores_)

        self.best_lambda_ = self.lambda_values[best_idx]
        self.beta_ = self.betas_[best_idx]
        self.intercept_ = self.intercepts_[best_idx]

    
    def validate(self, X_valid, y_valid, measure='f1'):
        """
        Compute validation score using the selected metric.

        Parameters
        ----------
        X_valid : array-like
            Validation features.
        y_valid : array-like
            Validation labels.
        measure : str
            Evaluation metric.

        Returns
        -------
        float
            Computed validation score.
        """

        proba = 1 / (1 + np.exp(-(X_valid @ self.beta_ + self.intercept_)))
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
        Run FISTA optimization for logistic loss with L1 regularization.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        lambda_value : float
            Regularization strength.

        Returns
        -------
        tuple
            Optimized coefficients and intercept.
        """

        beta = np.zeros(X_train.shape[1])
        intercept = 0.0

        y_beta = beta.copy()
        y_intercept = intercept

        t = 1
        L = np.linalg.norm(X_train, 2)**2 / 4
        
        for _ in range(self.max_iter):

            beta_old = beta.copy()
            intercept_old = intercept

            grad_beta, grad_intercept = self._gradient(X_train, y_train, y_beta, y_intercept)
            beta = self._soft_thresholding(y_beta - grad_beta / L, lambda_value / L)

            if self.fit_intercept:
                intercept = y_intercept - grad_intercept / L

            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y_beta = beta + ((t - 1) / t_new) * (beta - beta_old)
            
            if self.fit_intercept:
                y_intercept = intercept + ((t - 1) / t_new) * (intercept - intercept_old)
        
            t = t_new

            diff = np.linalg.norm(beta - beta_old)
            if self.fit_intercept:
                diff += abs(intercept - intercept_old)
            if diff < self.tol:
                break

        return beta, intercept
        
        
    
    def predict_proba(self, X_test):
        """
        Return predicted probabilities for input samples.

        Parameters
        ----------
        X_test : array-like
            Input features.

        Returns
        -------
        array-like
            Predicted probabilities.
        """

        if self.beta_ is None:
            raise ValueError("Model not trained. Run fit() first.")

        z = X_test @ self.beta_

        if self.fit_intercept:
            z += self.intercept_

        return 1 / (1 + np.exp(-z))
    
    
    def predict(self, X_test, threshold=0.5):
        """
        Return binary predictions using a given threshold.

        Parameters
        ----------
        X_test : array-like
            Input features.
        threshold : float
            Decision threshold.

        Returns
        -------
        array-like
            Predicted class labels.
        """
        return (self.predict_proba(X_test) >= threshold).astype(int)


    def plot(self):
        """
        Plot validation scores across lambda values.
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
        Plot coefficient paths as a function of lambda.
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


    def _gradient(self, X, y, beta, intercept):
        """
        Compute gradients of logistic loss.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Target labels.
        beta : array-like
            Model coefficients.
        intercept : float
            Model intercept.

        Returns
        -------
        tuple
            Gradients for coefficients and intercept.
        """
        Xb = X @ beta
        
        if self.fit_intercept:
            Xb += intercept

        z = 1 / (1 + np.exp(-Xb))
        error = z - y

        grad_beta = (X.T @ error)

        if self.fit_intercept:
            grad_intercept = np.sum(error)
        else:
            grad_intercept = 0.0

        return grad_beta, grad_intercept
    

    def _soft_thresholding(self, v, alpha):
        """
        Apply the soft-thresholding operator for L1 regularization.
        Parameters
        ----------
        v : array-like
            Input vector.
        alpha : float
            Threshold value.

        Returns
        -------
        array-like
            Thresholded vector.
        """
        return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)
