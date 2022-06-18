import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

ITERATIONS = 1000


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def plot_convergence_rate(values: List,
                          title: str = "",
                          color: str = "blue",
                          iterations=1000) -> go.Figure:
    """
    Plot the convergence rate of the gradient descent algorithm

    Parameters:
    -----------
    values: np.ndarray
        Loss values of the currently learned function.

    title: str, default=""
        Setting details to add to plot title

    iter_range: int, default=1000
        Plot's x-axis range (number of iterations)

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's convergence rate.
    """

    return go.Figure(
        [go.Scatter(x=list(range(iterations)), y=values,
                    mode="markers", marker_color=color)],
        layout=go.Layout(title=f"GD Convergence Rate {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights_lst = [], []

    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        weights_lst.append(weights)
        values.append(val)

    return callback, values, weights_lst


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        learning_rate = FixedLR(base_lr=eta)
        l1 = L1(init.copy())
        l2 = L2(init.copy())

        callback_l1, vals_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, vals_l2, weights_l2 = get_gd_state_recorder_callback()
        weights_l1.insert(0, init)
        weights_l2.insert(0, init)
        vals_l1.insert(0, l1.compute_output())
        vals_l2.insert(0, l2.compute_output())
        GradientDescent(learning_rate=learning_rate, callback=callback_l1,
                        max_iter=ITERATIONS).fit(
            f=l1, X=None, y=None)
        GradientDescent(learning_rate=learning_rate, callback=callback_l2,
                        max_iter=ITERATIONS).fit(
            f=l2, X=None, y=None)

        fig_l1 = plot_descent_path(module=L1,
                                   descent_path=np.array(weights_l1),
                                   title=f"L1, \u03B7 = {eta}")
        fig_l2 = plot_descent_path(module=L2,
                                   descent_path=np.array(weights_l2),
                                   title=f"L2, \u03B7 = {eta}")
        fig_l1.show(renderer="browser")
        fig_l2.show(renderer="browser")

        fig_l1 = plot_convergence_rate(values=vals_l1,
                                       title=f"L1, \u03B7 = {eta}",
                                       color="blue")
        fig_l2 = plot_convergence_rate(values=vals_l2,
                                       title=f"L2, \u03B7 = {eta}",
                                       color="orange")
        fig_l1.show(renderer="browser")
        fig_l2.show(renderer="browser")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    for gamma in gammas:
        learning_rate = ExponentialLR(base_lr=eta, decay_rate=gamma)

        l1 = L1(init.copy())

        callback_l1, vals_l1, weights_l1 = get_gd_state_recorder_callback()
        weights_l1.insert(0, init)
        vals_l1.insert(0, l1.compute_output())
        GradientDescent(learning_rate=learning_rate, callback=callback_l1,
                        max_iter=ITERATIONS).fit(
            f=l1, X=None, y=None)

        # Plot algorithm's convergence for the different values of gamma
        fig_l1 = plot_convergence_rate(values=vals_l1,
                                       title=f"L1, Exponential Learning Rate: "
                                             f"\u03B7={eta}, \u03B3={gamma}",
                                       color="blue")
        fig_l1.show(renderer="browser")

        # Plot descent path for gamma=0.95
        if gamma == 0.95:
            fig = plot_descent_path(module=L1,
                                    descent_path=np.array(weights_l1),
                                    title=f"GD Descent Path L1, Exponential "
                                          f"Learning Rate: \u03B7={eta}, "
                                          f"\u03B3={gamma}")
            fig.show(renderer="browser")


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def plot_roc_curve(y, y_prob):
    from sklearn.metrics import roc_curve, auc
    from utils import custom

    c = [custom[0], custom[-1]]

    fpr, tpr, thresholds = roc_curve(y, y_prob)

    return go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # # Plotting convergence rate of logistic regression over SA heart disease data
    # raise NotImplementedError()
    #
    # # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # # of regularization parameter
    # raise NotImplementedError()

    module = LogisticRegression(include_intercept=True,
                                solver=GradientDescent(),
                                penalty="l1")
    module.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = module.predict(X_train.to_numpy())
    y_prob = module.predict_proba(X_train.to_numpy())
    plot_roc_curve(y_pred, y_prob).show()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
