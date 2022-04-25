from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, nk = np.unique(y, return_counts=True)
        self.pi_ = nk / y.shape[0]

        x_sums = np.zeros((self.classes_.shape[0], X.shape[1]))
        for idx, val in enumerate(y):
            i = np.where(self.classes_ == val)[0]
            x_sums[i] += X[idx]
        self.mu_ = x_sums / nk[:, None]

        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i, k in enumerate(self.classes_):
            self.vars_[i] = np.diag((X[y == k] - self.mu_[i, :]).T
                                    @ (X[y == k] - self.mu_[i, :]))
        self.vars_ /= nk[:, None]

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

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

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
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihoods = []
        m, d = X.shape

        def normal_distribution_per_class(x):
            x_likelihoods = []
            for i, k in enumerate(self.classes_):
                a = 1 / np.sqrt(((2 * np.pi) ** d) * np.prod(self.vars_[i]))
                b = np.exp(-0.5 * (x - self.mu_[i]) * (1 / self.vars_[i]) @ (
                            x - self.mu_[i]).T)
                x_likelihoods.append(a * b * self.pi_[i])
            likelihoods.append(x_likelihoods)

        np.apply_along_axis(normal_distribution_per_class, arr=X, axis=1)
        return np.array(likelihoods)

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
