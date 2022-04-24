from typing import NoReturn

import numpy

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, nk = np.unique(y, return_counts=True)
        m = y.shape[0]
        d = X.shape[1]
        self.pi_ = nk / m
        x_sums = np.zeros((self.classes_.shape[0], d))
        for idx, val in enumerate(y):
            i = np.where(self.classes_ == val)[0]
            x_sums[i] += X[idx]
        self.mu_ = x_sums / nk[:, None]
        mus = np.array([self.mu_[np.where(self.classes_ == val)[0]] for val in y]).reshape(m, d)
        self.cov_ = ((X - mus).T @ (X - mus)) / m
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a = (self._cov_inv @ self.mu_.T)
        arg = np.array([mu @ self._cov_inv @ mu.T for mu in self.mu_])
        b = np.log(self.pi_) - arg / 2
        indexes = np.argmax(a.T @ X.T + b[:, None], axis=0)
        return np.array([self.classes_[i] for i in indexes])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        def multivariate_gaussian_pdf(x):
            exp_power = -0.5 * (x - self.mu_).T @ self._cov_inv @ (
                x - self.mu_)
            return 1 / (np.sqrt(((2 * np.pi) ** d)) * det(self.cov_)) * np.exp(
                   exp_power)

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m, d = X.shape
        k = len(self.classes_)
        return np.ndarray([multivariate_gaussian_pdf(x) * self.pi_[i]
                           for x in X
                           for i in range(k)]).reshape(m, k)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
