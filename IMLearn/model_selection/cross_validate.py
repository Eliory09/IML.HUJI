from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = 0, 0

    X_cvs = np.array_split(X, cv)
    y_cvs = np.array_split(y, cv)

    for i in range(cv):
        X_train = np.concatenate(X_cvs[:i] + X_cvs[i + 1:])
        y_train = np.concatenate(y_cvs[:i] + y_cvs[i + 1:])
        estimator.fit(X_train, y_train)

        train_pred = estimator.predict(X_train)
        val_pred = estimator.predict(X_cvs[i])
        train_score += scoring(y_train, train_pred)
        validation_score += scoring(y_cvs[i], val_pred)

    train_score /= cv
    validation_score /= cv

    return train_score, validation_score
