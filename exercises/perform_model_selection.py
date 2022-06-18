from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    low, high = -1.2, 2
    X = np.linspace(low, high, n_samples)
    y_noiseless = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y_noise = y_noiseless + np.random.normal(0, np.sqrt(noise), n_samples)
    train_proportions = 2 / 3
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y_noise),
                                                        train_proportions)
    X_train, y_train, X_test, y_test = \
        np.asarray(X_train).flatten(), \
        np.asarray(y_train).flatten(), \
        np.asarray(X_test).flatten(), \
        np.asarray(y_test).flatten()
    fig = go.Figure([go.Scatter(x=X, y=y_noiseless,
                                name=r"Noiseless Data",
                                mode='markers',
                                marker=dict(color="black"),
                                showlegend=False),
                     go.Scatter(x=X_train, y=y_train,
                                name=r"Noised Train Data",
                                mode='markers',
                                marker=dict(color="blue"),
                                showlegend=False),
                     go.Scatter(x=X_test, y=y_test,
                                name=r"Noised Test Data",
                                mode='markers',
                                marker=dict(color="orange"),
                                showlegend=False)],
                    layout=dict(
                        title=r"$\textbf{(1) True Dataset "
                              r"and Noised Dataset}$"))
    fig.show(renderer="browser")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, validation_errors = [], []
    degrees = np.linspace(0, 9, 10).astype(int)
    for k in degrees:
        estimator = PolynomialFitting(k)
        train_err, validation_err = cross_validate(estimator, X_train,
                                                   y_train,
                                                   scoring=mean_square_error)
        train_errors.append(train_err)
        validation_errors.append(validation_err)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=degrees, y=train_errors, name=r"Train Errors"))
    fig.add_trace(
        go.Bar(x=degrees, y=validation_errors, name=r"Validation Errors")
    )
    fig.update_layout(
        title=r"$\textbf{(2) 5-folds Cross Validation"
              r"Train Errors and Test Errors}$",
        barmode="group", xaxis_title="Polynomial degree",
        yaxis_title="MSE"
    )
    fig.show(renderer="browser")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(validation_errors)
    model = PolynomialFitting(k)
    model.fit(X_train, y_train)
    mse = mean_square_error(y_test, model.predict(X_test))
    print(f"k={k}, MSE={np.round(mse, 2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[: n_samples], y[: n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    low, high = 0, 3
    lambda_range = np.linspace(low, high, n_evaluations)

    ridge_train_errors, ridge_validation_errors = [], []
    lasso_train_errors, lasso_validation_errors = [], []

    for lam in lambda_range:
        ridge = RidgeRegression(lam)
        ridge_train_err, ridge_validation_err = cross_validate(ridge, X_train,
                                                               y_train,
                                                               mean_square_error)
        ridge_train_errors.append(ridge_train_err)
        ridge_validation_errors.append(ridge_validation_err)

        lasso = Lasso(lam)
        lasso_train_err, lasso_validation_err = cross_validate(lasso,
                                                               np.asarray(
                                                                   X_train),
                                                               np.asarray(
                                                                   y_train),
                                                               mean_square_error)
        lasso_train_errors.append(lasso_train_err)
        lasso_validation_errors.append(lasso_validation_err)

    fig = go.Figure([go.Scatter(x=lambda_range, y=ridge_validation_errors,
                                name="Ridge Validation Errors",
                                mode="lines",
                                line=dict(color="orange", width=2)),
                     go.Scatter(x=lambda_range, y=lasso_validation_errors,
                                name="Lasso Validation Errors",
                                mode="lines",
                                line=dict(color="red", width=2)),
                     go.Scatter(x=lambda_range, y=ridge_train_errors,
                                name="Ridge Train Errors",
                                mode="lines",
                                line=dict(color="blue", width=2)),
                     go.Scatter(x=lambda_range, y=lasso_train_errors,
                                name="Lasso Train Errors",
                                mode="lines",
                                line=dict(color="green", width=2))],
                    layout=dict(
                        title=rf"$\textbf{{(1) 5-folds Cross Valiation for 0-3 "
                              rf"Regularization Parameters Errors for Ridge and Lasso Regressors}}$"))
    fig.show(renderer="browser")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    # Extract best lambdas
    ridge_lam = lambda_range[np.argmin(ridge_validation_errors)]
    lasso_lam = lambda_range[np.argmin(lasso_validation_errors)]
    ridge = RidgeRegression(lam=ridge_lam)
    lasso = Lasso(alpha=lasso_lam)
    linear_regression = LinearRegression()

    # Fit the models
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    linear_regression.fit(X_train, y_train)

    # Calculate the test errors for each model
    ridge_err = ridge.loss(X_test, y_test)
    lasso_err = mean_square_error(lasso.predict(X_test), y_test)
    linear_regression_err = linear_regression.loss(X_test, y_test)

    print(
        f"Test error of ridge regression with lambda = {ridge_lam}: {ridge_err}")
    print(
        f"Test error of lasso regression with lambda = {lasso_lam}: {lasso_err}")
    print(
        f"Test error of linear regression = {linear_regression_err}")


if __name__ == '__main__':
    np.random.seed(0)
select_polynomial_degree()
select_polynomial_degree(noise=0)
select_polynomial_degree(n_samples=1500, noise=10)
select_regularization_parameter()
