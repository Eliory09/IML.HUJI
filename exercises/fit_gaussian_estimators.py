from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, size=1000)
    uv_gaussian = UnivariateGaussian()
    uv_gaussian.fit(X)
    print((uv_gaussian.mu_, uv_gaussian.var_))

    # # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    ms = np.linspace(10, 1000, 100).astype(int)
    for m in ms:
        estimated_mean.append(UnivariateGaussian().fit(X[:m]).mu_)

    go.Figure([go.Scatter(x=ms, y=np.abs(np.array(estimated_mean) - mu),
                          mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Absolute distance between the estimated "
                        r"and true value of the expectation, as a function "
                        r"of the sample size}$",
                  xaxis=dict(title="$m\\text{ - number of samples}$"),
                  yaxis=dict(title="r$|\\hat\\mu-\\mu|$"))).show(
        renderer="browser")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uv_gaussian.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf,
                          mode='markers',
                          name=r'$PDF$')],
              layout=go.Layout(
                  title=r"$\text{Empirical PDF function under the fitted "
                        r"normal-distributed sample}$",
                  xaxis=dict(title="$\\text{Sample values}$"),
                  yaxis=dict(title="PDF"))).show(
        renderer="browser")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    mv_gaussian = MultivariateGaussian()
    mv_gaussian.fit(X)
    print(mv_gaussian.mu_)
    print(mv_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    ll_mat = np.zeros((200, 200))
    ms = np.linspace(-10, 10, 200)
    for i, f1 in enumerate(ms):
        for j, f3 in enumerate(ms):
            mu = np.array([f1, 0, f3, 0])
            ll_mat[i, j] = mv_gaussian.log_likelihood(mu, cov, X)

    go.Figure([go.Heatmap(x=ms, y=ms, z=ll_mat,
                          type='heatmap',
                          colorscale='Viridis')],
              layout=go.Layout(
                  title=r"$\text{Log-likelihood value of X~} \mathcal{N}(\mu,"
                        r"\Sigma), \mu = [f1, 0, f3, 0]^{T}$",
                  xaxis=dict(title=r"$f3$"),
                  yaxis=dict(title=r"$f1$"))).show(renderer="browser")

    # # Question 6 - Maximum likelihood
    idx = np.unravel_index(np.argmax(ll_mat, axis=None), ll_mat.shape)
    print("f1 = {:.3f}".format(ms[idx[0]]))
    print("f3 = {:.3f}".format(ms[idx[1]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
