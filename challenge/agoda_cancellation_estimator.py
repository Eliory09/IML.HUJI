from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import re
from datetime import datetime

import sklearn.model_selection

from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.features = None
        self.labels = None
        self.model = RandomForestClassifier()
        self.top_5_countries = []
        self.hotels_top_5_countries = []
        self.hotels_top_5_cities = []

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        train_X, test_X, train_y, test_y = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=0.1)
        self.model.fit(train_X, train_y)
        print(self.model.score(test_X, test_y))

        self.model.fit(X, y)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pass

    def load_data(self, filename: str, url: str):
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
        booking_checkin_diff = [int(diff.days) for diff in
                                booking_checkin_diff]
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

        ser = pd.Series(has_cancelled)
        features.insert(2, 'has_cancelled', ser)

        cancellation_days_before_checkin = []
        for i, cancel_date in enumerate(features['cancellation_datetime']):
            if cancel_date is np.nan:
                cancellation_days_before_checkin.append(np.nan)
                continue
            cancel_date = datetime.strptime(cancel_date, "%Y-%m-%d").date()
            cancellation_days_before_checkin.append(
                int((features['checkin_date'][i] - cancel_date).days))
        ser = pd.Series(cancellation_days_before_checkin)
        features.insert(3, 'cancellation_days_before_checkin', ser)

        features = pd.get_dummies(features, columns=[
            'hotel_star_rating',
            'accommadation_type_name',
            'charge_option',
            'original_payment_type',
        ])

        vacation_duration = (
                features['checkout_date'] - features['checkin_date'])
        vacation_duration = [int(diff.days) for diff in vacation_duration]
        features.insert(4, 'vacation_duration', vacation_duration)
        usd_prices = [
            amount / currencies_data[features['original_payment_currency'][i]]
            for
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
                    amount = int(x.group(4)) / features['vacation_duration'][
                        i] * \
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

        features['is_user_logged_in'] = [
            0 if val == 'False' or not val or val is np.nan else 1
            for val in features['is_user_logged_in']]

        features['is_first_booking'] = [
            0 if val == 'False' or not val or val is np.nan else 1
            for val in features['is_first_booking']]

        for cat in ['request_nonesmoke', 'request_latecheckin',
                    'request_highfloor',
                    'request_largebed', 'request_twinbeds',
                    'request_earlycheckin', 'request_airport', "cancellation_days_before_checkin"]:
            features[cat] = features[cat].fillna(0)

        labels = features["has_cancelled"]

        # Origin country code
        group = (features.groupby(['origin_country_code'])[
                     'has_cancelled'].sum()
                 / features.groupby(
                    ['origin_country_code']).size()).reset_index(name="ratio")
        group = group.sort_values(by=['ratio'], ascending=False).head(5)
        top_5_countries = \
            [1 if code in list(group['origin_country_code']) else 0
             for code in features['origin_country_code']]
        features.insert(7, 'origin_top_5_countries', top_5_countries)
        self.top_5_countries = list(group['origin_country_code'])

        # Hotel country code
        group = (features.groupby(['hotel_country_code'])[
                     'has_cancelled'].sum()
                 / features.groupby(
                    ['hotel_country_code']).size()).reset_index(
            name="ratio")
        group = group.sort_values(by=['ratio'], ascending=False).head(5)
        hotels_top_5_countries = \
            [1 if code in list(group['hotel_country_code']) else 0
             for code in features['hotel_country_code']]
        features.insert(7, 'hotel_top_5_countries', hotels_top_5_countries)
        self.hotels_top_5_countries = list(group['hotel_country_code'])

        # Hotel city code
        group = (features.groupby(['hotel_city_code'])['has_cancelled'].sum()
                 / features.groupby(['hotel_city_code']).size()).reset_index(
            name="ratio")

        group = group.sort_values(by=['ratio'], ascending=False).head(5)
        hotel_top_5_cities = \
            [1 if code in list(group['hotel_city_code']) else 0
             for code in features['hotel_city_code']]
        features.insert(7, 'hotel_top_5_cities', hotel_top_5_cities)
        self.hotels_top_5_cities = list(group['hotel_city_code'])


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
            'hotel_area_code',
            "has_cancelled",
            "hotel_country_code",
            "hotel_city_code",
            "origin_country_code"
        ], axis=1, inplace=True)

        self.features = features
        self.labels = labels
        return features, labels
