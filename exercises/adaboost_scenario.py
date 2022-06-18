import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_errs, test_errs = [], []
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    for t in range(1, n_learners + 1):
        train_errs.append(model.partial_loss(train_X, train_y, t))
        test_errs.append(model.partial_loss(test_X, test_y, t))

    fig = go.Figure([go.Scatter(x=list(range(1, n_learners + 1)), y=train_errs,
                                name=r"Train Set",
                                mode='lines',
                                line=dict(color="blue"),
                                showlegend=False),
                     go.Scatter(x=list(range(1, n_learners + 1)), y=test_errs,
                                name=r"Test Set",
                                mode='lines',
                                line=dict(color="orange"),
                                showlegend=False)],
                    layout=dict(
                        title=r"$\textbf{(1) Misclassification Error "
                              r"as a Function of Learners Number}$"))
    fig.show(renderer="browser")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = {1: "circle", -1: "x"}
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        rf"$\textbf{{(1) 5 Iterations}}$",
        rf"$\textbf{{(2) 50 Iterations}}$",
        rf"$\textbf{{(3) 100 Iterations}}$",
        rf"$\textbf{{(4) 250 Iterations}}$"))

    X = test_X
    for i, t in enumerate(T):
        y = model.partial_predict(X, t)
        graph = decision_surface(
            lambda X: model.partial_predict(X, t), lims[0], lims[1],
            showscale=False)
        fig.add_trace(graph, row=1 + int(i / 2), col=1 + i % 2)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=y, symbol=[symbols[y_i] for y_i in y],
                                   colorscale=[custom[0], custom[-1]]),
                       line=dict(color="black", width=1))
            , row=1 + i // 2, col=1 + i % 2)

    fig.update_layout(
        title=rf"$\textbf{{(2) Decision Boundaries Of "
              rf"Decision Stumps Ensemble}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show(renderer="browser")

    # Question 3: Decision surface of best performing ensemble
    t = np.argmin(test_errs) + 1
    y = model.partial_predict(X, int(t))
    acc = accuracy(test_y, y)
    fig = go.Figure([
        decision_surface(
            lambda X: model.partial_predict(X, t), lims[0], lims[1],
            showscale=False),
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=y, symbol=[symbols[y_i] for y_i in y],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))],
        layout=dict(
            title=rf"$\textbf{{ (3) Decision Boundaries Of Best "
                  rf"Performing Ensemble. Size={t}, Accuracy={float(str(acc)[0: 4])} }}$"))
    fig.show(renderer="browser")


    # Question 4: Decision surface with weighted samples
    t = n_learners
    y = model.partial_predict(train_X, n_learners)
    D = model.D_ / np.max(model.D_) * 5
    fig = go.Figure([
        decision_surface(
            lambda X: model.partial_predict(X, t), lims[0], lims[1], t,
            showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                   showlegend=False,
                   marker=dict(color=y, symbol=[symbols[y_i] for y_i in y],
                               size=D,
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))],
        layout=dict(
            title=rf"$\textbf{{ (4) Decision Boundaries Of {t} Learners "
                  rf"Ensemble With Weighted Samples }}$"))
    fig.show(renderer="browser")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, n_learners=250, train_size=5000,
                              test_size=500)
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250, train_size=5000,
                              test_size=500)
