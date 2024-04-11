import pickle
import inflection
import pandas as pd
import numpy as np
import datetime


class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df1):
        # Rename columns to work better further in the project

        cols_old = df1.columns
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new

        # Data Types

        df1['date'] = pd.to_datetime(df1['date'])

        # Fillout NA

        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 50000 if pd.isna(x) else x)
        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['date'].month if pd.isna(x['competition_open_since_month'])
            else x['competition_open_since_month'], axis=1)
        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['competition_open_since_year'])
            else x['competition_open_since_year'], axis=1)
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if pd.isna(x['promo2_since_week'])
        else x['promo2_since_week'], axis=1)
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if pd.isna(x['promo2_since_year'])
        else x['promo2_since_year'], axis=1)
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep',
                     10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df1['promo_interval'] = df1['promo_interval'].fillna(0)
        df1['month_name'] = df1['date'].dt.strftime('%b')
        df1['is_promo'] = df1.apply(
            lambda x: 1 if x['promo_interval'] != 0 and x['month_name'] in x['promo_interval'].split(',') else 0,
            axis=1)

        # Change Data Types

        df1 = df1.astype({'competition_open_since_month': int,
                          'competition_open_since_year': int,
                          'promo2_since_week': int,
                          'promo2_since_year': int})
        return df1

    def feature_engineering(self, df2):
        # Adding new features

        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')
        df2['competition_since'] = pd.to_datetime(
            df2['competition_open_since_year'].astype(str) + '-' + df2['competition_open_since_month'].astype(str))
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days).astype(
            int)
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']) / 7).apply(lambda x: x.days).astype(int)
        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        df2['state_holiday'] = df2['state_holiday'].map(
            {'a': 'public_holiday', 'b': 'easter_holiday', 'c': 'christmas', '0': 'regular_day'})

        # Filtering

        df2 = df2[df2['open'] != 0]
        drop_columns = ['open', 'month_name', 'promo_interval']
        df2 = df2.drop(drop_columns, axis=1)

        return df2

    def data_preparation(self, df5):
        # Rescaling

        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(
            df5[['competition_distance']].values)
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(
            df5[['competition_time_month']].values)
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        # Encoding

        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # Nature Transformation ( Apply cyclical transformations )

        def encode_cyclical_feature(df, feature, period):
            df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period)
            df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period)

        encode_cyclical_feature(df5, 'day_of_week', 7)
        encode_cyclical_feature(df5, 'month', 12)
        encode_cyclical_feature(df5, 'day', 30)
        encode_cyclical_feature(df5, 'week_of_year', 52)

        cols_selected = ['store', 'promo', 'store_type', 'competition_distance',
                         'competition_open_since_month', 'competition_open_since_year', 'promo2_since_week',
                         'competition_time_month', 'promo_time_week', 'day_of_week_sin',
                         'day_of_week_cos', 'month_cos', 'day_cos']

        return df5[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')
