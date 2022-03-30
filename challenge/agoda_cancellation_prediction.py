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


def load_data(filename: str, url: str):
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
    week_start = datetime(2018, 5, 30).date()
    week_end = datetime(2018, 6, 6).date()

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
    # has_cancelled = [0 if date is np.nan else 1
    #                  for date in features['cancellation_datetime']]
    has_cancelled = []
    for date in features['cancellation_datetime']:
        if date is np.nan:
            has_cancelled.append(0)
            continue
        date = datetime.strptime(date, "%Y-%m-%d").date()
        is_in_week = 1 if week_start <= date <= week_end else 0
        has_cancelled.append(is_in_week)

    # cancellation_days_before_checkin = []
    # for i, cancel_date in enumerate(features['cancellation_datetime']):
    #     if cancel_date is np.nan:
    #         cancellation_days_before_checkin.append(np.nan)
    #         continue
    #     cancellation_days_before_checkin.append(
    #         int((cancel_date - features['checkin_date'][i]).days))
    # ser = pd.Series(cancellation_days_before_checkin)
    # features.insert(3, 'cancellation_days_before_checkin', ser)

    ser = pd.Series(has_cancelled)
    features.insert(2, 'has_cancelled', ser)
    features = pd.get_dummies(features, columns=[
        'hotel_country_code',
        'hotel_star_rating',
        'accommadation_type_name',
        'charge_option',
        'origin_country_code',
        'original_payment_type',
        'hotel_city_code'])

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
    features.drop([
        'h_booking_id',
        'hotel_id',
        'booking_datetime',
        'checkin_date',
        'checkout_date',
        'cancellation_datetime',
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
        'hotel_area_code'
    ], axis=1, inplace=True)
    labels = full_data["cancellation_datetime"]

    return features, labels


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
    # Load data
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    df, cancellation_labels = load_data(
        "../datasets/agoda_cancellation_train.csv", url)
    train_X, train_y, test_X, test_y = train_test_split(df,
                                                        cancellation_labels)

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
