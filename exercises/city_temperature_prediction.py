import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=True).dropna().drop_duplicates()
    data.drop(data[data.Temp < -40].index, inplace=True)
    data.Date = [date.date() for date in pd.to_datetime(data.Date)]
    data["DayOfYear"] = [date.timetuple().tm_yday for date in data.Date]
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    isr_data = df[df.Country == "Israel"]
    years = isr_data["Year"].astype(str)
    fig1 = px.scatter(isr_data, x='DayOfYear', y='Temp',
                     color=years)
    fig1.show(renderer="browser")

    isr_data_by_month = isr_data.groupby('Month').Temp.agg('std').reset_index()
    fig2 = px.bar(isr_data_by_month, x='Month', y='Temp')
    fig2.update_layout(dict(
        title="Std of Temp. as a Function of the Month in Israel: 1995-2007"
    ))
    fig2.update_xaxes(dict(title='Month', tickmode='linear'))
    fig2.update_yaxes(dict(title='Std of Temp.'))
    fig2.show(renderer="browser")

    # Question 3 - Exploring differences between countries
    dat = df.groupby(['Country', 'Month']).Temp.agg(
        ['mean', 'std']).reset_index()
    fig3 = px.line(dat, x='Month', y='mean', error_y='std', color='Country')
    fig3.update_layout(dict(
        title="Average of Temp. as a Function of the Month"
    ))
    fig3.update_xaxes(dict(title='Month', tickmode='linear'))
    fig3.update_yaxes(dict(title='Average of Temp.'))
    fig3.show(renderer="browser")

    # Question 4 - Fitting model for different values of `k`
    y = isr_data['Temp']
    X = isr_data['DayOfYear']

    train_X, train_y, test_X, test_y = split_train_test(X, y,
                                                        train_proportion=0.75)

    loss_lst = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = round(model.loss(test_X.to_numpy(),
                             test_y.to_numpy()), 2)
        loss_lst.append(loss)
        print(loss)

    fig4 = px.bar(loss_lst, x=range(1, 11), y=loss_lst)
    fig4.update_layout(dict(
        title="Test Error as a Function of the Polynomial Degree"
    ))
    fig4.update_traces(marker_color='cadetblue')
    fig4.update_xaxes(dict(title='k - Degree', tickmode='linear'))
    fig4.update_yaxes(dict(title='MSE'))
    fig4.show(renderer="browser")

    # # Question 5 - Evaluating fitted model on different countries
    best_k = 3
    model = PolynomialFitting(best_k)
    model.fit(train_X.to_numpy(), train_y.to_numpy())
    loss_lst = []
    countries = list(df.Country.unique())
    countries.remove("Israel")
    for country in countries:
        y = df[df['Country'] == country]['Temp']
        X = df[df['Country'] == country]['DayOfYear']
        loss_lst.append(
            round(model.loss(X.to_numpy(),
                             y.to_numpy()), 2))

    fig5 = px.bar(loss_lst, x=countries, y=loss_lst)
    fig5.update_layout(dict(
        title="Test Error by Countries, Degree = 3"
    ))
    fig5.update_traces(marker_color='cadetblue')
    fig5.update_xaxes(dict(title='Country', tickmode='linear'))
    fig5.update_yaxes(dict(title='MSE'))
    fig5.show(renderer="browser")

