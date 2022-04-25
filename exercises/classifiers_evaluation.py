from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        X_ = data[:, :2]
        y_ = data[:, 2]
        model = Perceptron(callback=lambda obj, x, y: losses.append(
            obj.loss(X_, y_))).fit(X_, y_)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure([go.Scatter(x=np.linspace(1, 1000, 1000), y=losses,
                                    name=r"Misclassification Error over 1000 "
                                         r"iterations of Perceptron Algorithm "
                                         r"on dataset",
                                    mode='markers+lines',
                                    marker=dict(color="#5BC5E5"),
                                    line=dict(color="#5BC5E5", dash='dash'),
                                    showlegend=False)],
                        layout=dict(
                            title=r"Misclassification Error over 1000 "
                                  r"iterations of Perceptron Algorithm "
                                  r"on dataset"))
        fig.show(renderer="browser")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "x", "square"])

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_y_pred = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_y_pred = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # Add traces for data-points setting symbols and colors
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            rf"$\textbf{{(1) {f} Dataset, "
            rf"Classifier: LDA, "
            rf"Accuracy: {accuracy(y, lda_y_pred)}}}$",
            rf"$\textbf{{(1) {f} Dataset, "
            rf"Classifier: Gaussian Naive Bayes, "
            rf"Accuracy: {np.round(accuracy(y, gnb_y_pred), 2)}}}$"))

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=lda_y_pred, symbol=symbols[y],
                                   line=dict(color="black", width=1),
                                   colorscale="plasma")),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=gnb_y_pred, symbol=symbols[y],
                                   line=dict(color="black", width=1),
                                   colorscale="plasma")),
            row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x", size=20)),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x", size=20)),
            row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
