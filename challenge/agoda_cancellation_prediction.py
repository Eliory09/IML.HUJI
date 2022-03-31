import re
from datetime import datetime

import sklearn.model_selection

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def load_data(filename: str, url: str, estimator: AgodaCancellationEstimator):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    currencies_data = requests.get(url).json()['rates']
    week_start = datetime(2018, 12, 7).date()
    week_end = datetime(2018, 12, 13).date()

    features = full_data
    features['booking_datetime'] = \
        [datetime.strptime(date[0:10], "%Y-%m-%d").date() for date in
         features['booking_datetime']]
    features['checkin_date'] = \
        [datetime.strptime(date[0:10], "%Y-%m-%d").date() for date in
         features['checkin_date']]
    features['checkout_date'] = \
        [datetime.strptime(date[0:10], "%Y-%m-%d").date() for date in
         features['checkout_date']]
    booking_checkin_diff = (
            features['checkin_date'] - features['booking_datetime'])
    booking_checkin_diff = [int(diff.days) for diff in booking_checkin_diff]
    features.insert(1, 'booking_checkin_diff', booking_checkin_diff)

    cancellation_days_before_checkin = []
    for i, checkin_date in enumerate(features['checkin_date']):
        cancellation_days_before_checkin.append(
            int((checkin_date - week_start).days))
    ser = pd.Series(cancellation_days_before_checkin)
    features.insert(3, 'cancellation_days_before_checkin', ser)

    features = pd.get_dummies(features, columns=[
        # 'hotel_country_code',
        'hotel_star_rating',
        'accommadation_type_name',
        'charge_option',
        # 'origin_country_code',
        'original_payment_type',
        # 'hotel_city_code'
    ])

    vacation_duration = (
            features['checkout_date'] - features['checkin_date'])
    vacation_duration = [int(diff.days) for diff in vacation_duration]
    features.insert(4, 'vacation_duration', vacation_duration)
    usd_prices = [
        amount / currencies_data[features['original_payment_currency'][i]] for
        i, amount in enumerate(features['original_selling_amount'])]
    ser = pd.Series(usd_prices)
    features.insert(5, 'usd_prices', ser)

    POLICY_PHRASE = "(((\d+)D)*(100|[0-9][0-9]|[0-9])(P|N))+"

    policy_list = []
    for i, policy in enumerate(features['cancellation_policy_code']):
        policies = policy.split("_")
        if policy == "UNKNOWN" or policies is None:
            policy_list.append([])
            continue
        l = []
        for p in policies:
            x = re.search(POLICY_PHRASE, p)
            if x.group(3) is not None:
                days_before = int(x.group(3))
            else:
                days_before = np.nan
            if x.group(5) == 'P':
                amount = int(x.group(4)) / 100 * features['usd_prices'][i]
            else:
                amount = int(x.group(4)) / features['vacation_duration'][i] * \
                         features['usd_prices'][i]
            l.append([days_before, amount])
        policy_list.append(l)

    risks_in_usd = []
    for i, user_policies in enumerate(policy_list):
        days_before_checkin = int(
            (features['checkin_date'][i] - week_start).days)
        risk = 0
        for p in user_policies:
            if p[0] is np.nan:
                continue
            if days_before_checkin < p[0]:
                risk = p[1]
        risks_in_usd.append(risk)

    ser = pd.Series(risks_in_usd)
    features.insert(6, 'risks_in_usd', ser)
    # print(features['booking_checkin_diff'].corr(features['has_cancelled']))

    features['is_user_logged_in'] = [0 if val == 'False' or not val or val is np.nan else 1
                                     for val in features['is_user_logged_in']]

    features['is_first_booking'] = [
        0 if val == 'False' or not val or val is np.nan else 1
        for val in features['is_first_booking']]

    for cat in ['request_nonesmoke', 'request_latecheckin', 'request_highfloor',
                'request_largebed', 'request_twinbeds',
                'request_earlycheckin', 'request_airport', "cancellation_days_before_checkin"]:
        features[cat] = features[cat].fillna(0)

    top_5_countries = estimator.top_5_countries
    group = [1 if code in top_5_countries else 0
             for code in features['origin_country_code']]
    features.insert(7, 'origin_top_5_countries', group)

    hotels_top_5_countries = estimator.hotels_top_5_countries
    group = [1 if code in hotels_top_5_countries else 0
             for code in features['hotel_country_code']]
    features.insert(7, 'hotels_top_5_countries', group)

    hotels_top_5_cities = estimator.hotels_top_5_cities
    group = [1 if code in hotels_top_5_cities else 0
             for code in features['hotel_city_code']]
    features.insert(7, 'hotels_top_5_cities', group)

    features.drop([
        'h_booking_id',
        'hotel_id',
        'booking_datetime',
        'checkin_date',
        'checkout_date',
        # 'cancellation_datetime',
        'hotel_live_date',
        'h_customer_id',
        'customer_nationality',
        'guest_nationality_country_name',
        'language',
        'original_payment_method',
        'original_payment_currency',
        'original_selling_amount',
        'cancellation_policy_code',
        'hotel_brand_code',
        'hotel_chain_code',
        'hotel_area_code',
        # "has_cancelled",
        "hotel_country_code",
        "hotel_city_code",
        "origin_country_code"
    ], axis=1, inplace=True)

    return features


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)
    # Load data
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    estimator = AgodaCancellationEstimator()
    df, cancellation_labels = estimator.load_data(
        "../datasets/agoda_cancellation_train.csv", url)
    # fig, ax = go.subplots(figsize=(22, 15))
    # corr_df = df.corr()
    # go.Figure([go.Heatmap(x=df.columns, y=df.columns, z=corr_df,
    #                       type='heatmap',
    #                       colorscale='Viridis')]).show(renderer="browser")




    # Groupby origin_country_code

    # group = (df.groupby(['origin_country_code'])['has_cancelled'].sum()
    #          / df.groupby(['origin_country_code']).size()).reset_index(name="ratio")
    #
    # px.bar(group.sort_values(by=['ratio'], ascending=False).head(5), x="origin_country_code", y="ratio",
    #        height=500).show(renderer="browser")
    #
    # group1 = (df.groupby(['hotel_city_code'])[
    #              'has_cancelled'].sum() / df.groupby(
    #     ['hotel_city_code']).size()).reset_index(name="ratio")
    #
    # px.bar(group1.sort_values(by=['ratio'], ascending=False).head(5),
    #        x="hotel_city_code", y="ratio",
    #        height=500).show(renderer="browser")
    #
    # group2 = (df.groupby(['hotel_country_code'])[
    #               'has_cancelled'].sum() / df.groupby(
    #     ['hotel_country_code']).size()).reset_index(name="ratio")
    #
    # px.bar(group2.sort_values(by=['ratio'], ascending=False).head(5),
    #        x="hotel_country_code", y="ratio",
    #        height=500).show(renderer="browser")
    #
    # group.apply(lambda x: x['has_cancelled/count'] / x["origin_country_code/count"], axis=1)
    # group['has_cancelled'] = group['has_cancelled'] / group['size']

    # Fit model over data

    new_df = load_data("test_set_week_1.csv", url, estimator)

    # Get missing columns in the training test
    total_columns = set(df.columns).symmetric_difference(set(new_df.columns))
    # Add a missing column in test set with default value equal to 0
    for c in total_columns:
        if c not in new_df.columns:
            new_df[c] = 0
        if c not in df.columns:
            df[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    new_df = new_df[new_df.columns]
    df = df[df.columns]

    # Store model predictions over test set
    estimator.fit(df.to_numpy(), cancellation_labels)
    evaluate_and_export(estimator, new_df.to_numpy(), "313577207_316305101_dor.csv")
