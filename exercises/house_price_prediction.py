import datetime
import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Read .csv file, drop duplicates and drop NaN rows
    data = pd.read_csv(filename).dropna().drop_duplicates()

    # Fix minus values
    data.price = np.abs(data.price)
    data.sqft_lot15 = np.abs(data.sqft_lot15)

    # Drop empty row
    data = data.drop(data[(data.price == 0)].index)

    # Convert 'date' column into year and months dummy columns
    dates = data['date']
    dates = dates.apply(
        lambda date: datetime.datetime.strptime(date[0: 8], "%Y%m%d").date())
    years = dates.apply(lambda date: date.year)

    # Convert build and renovation years to a column
    # indicates the years past since last building renovation.
    # (2015 - max(Renovation_Year, 'Build_Year'))

    data['years_from_renovation'] = [
        years[index] - max(row['yr_renovated'], row['yr_built']) for index, row
        in
        data.iterrows()]

    # Convert columns into dummy columns
    data = pd.get_dummies(data, columns=[
        'zipcode'
    ])

    # Drop more extreme values.
    data.drop(data[data.sqft_lot > 120000].index, inplace=True)
    data.drop(data[data.bedrooms > 15].index, inplace=True)

    # Response vector
    resp = data.price

    # Drop unnecessary values (for now)
    data.drop([
        'date',
        'yr_renovated',
        'lat',
        'long',
        'id',
        'price'
    ], axis=1, inplace=True)

    return data, resp


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:
        x_ = X[feature]
        corr = np.cov(x_, y)[0][1] / (x_.std() * y.std())
        figure = go.Figure([
            go.Scatter(x=x_, y=y,
                       name=feature.capitalize() + r" as a Function of House "
                                      r"Prices, Pearson: " + str(round(corr, 3)),
                       mode='markers',
                       marker=dict(color="LightSkyBlue"),
                       showlegend=False)
        ],
            layout=dict(
                title=r"$\text{" + feature.capitalize() +
                      r" as a Function of House "
                      r"Prices, Pearson: " + str(round(corr, 3)) + "}$"))
        path = os.path.join(output_path, feature + ".png")
        figure.write_image(path)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, response = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, response, 'plots')

    # # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    x = np.linspace(10, 100)
    loss_var = []
    loss_mean = []
    for p in x:
        loss = []
        for _ in range(10):
            estimator = LinearRegression()
            train_sample_X = train_X.sample(frac=p / 100)
            train_sample_y = train_y[train_sample_X.index]
            estimator.fit(train_sample_X.to_numpy(), train_sample_y.to_numpy())
            loss.append(estimator.loss(test_X.to_numpy(), test_y.to_numpy()))
        loss_var.append(np.var(loss))
        loss_mean.append(np.mean(loss))

    fig = go.Figure([go.Scatter(x=x, y=loss_mean,
                                name=r"Mean Loss as a Function of P%",
                                mode='markers+lines',
                                marker=dict(color="red"),
                                line=dict(color="blue"),
                                showlegend=False),
                     go.Scatter(x=x,
                                y=np.array(loss_mean) - 2 * np.sqrt(loss_var),
                                fill=None,
                                mode="lines", line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=x,
                                y=np.array(loss_mean) + 2 * np.sqrt(loss_var),
                                fill='tonexty',
                                mode="lines", line=dict(color="lightgrey"),
                                showlegend=False)],
                    layout=dict(
                        title=r"$\text{(1) Mean Loss as a Function of P%}$"))
    fig.show(renderer="browser")
