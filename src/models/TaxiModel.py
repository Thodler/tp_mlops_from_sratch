import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

from utils.config_loader import load_config

config = load_config()

class TaxiModel:

    __TARGET_NAME = 'trip_duration'

    def __init__(self, model, set_train, set_test = None):
        self.test_processed = self.__preprocess(set_test)
        self.train_processed = self.__preprocess(set_train)
        self.model = model

    def __columnsFeature(self):
        num_features = ['abnormal_period', 'hour']
        cat_features = ['weekday', 'month']

        return num_features + cat_features

    def __preprocess(self, X):

        y =  X[self.__TARGET_NAME]
        X.drop(columns=[self.__TARGET_NAME])

        X.drop(columns=['id'], inplace=True)
        X.drop(columns=['dropoff_datetime'], inplace=True)
        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])


        X['pickup_date'] = X['pickup_datetime'].dt.date

        df_abnormal_dates = X.groupby('pickup_date').size()
        abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]

        X['weekday'] = X['pickup_datetime'].dt.weekday
        X['month'] = X['pickup_datetime'].dt.month
        X['hour'] = X['pickup_datetime'].dt.hour
        X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)

        return X, y

    def __postprocess(self, raw_output):
        # your postprocessing logic: inverse transformation, etc.
        # example: np.expm1(raw_output)
        return np.round(raw_output)

    def fit(self):
        y = self.train_processed[1]
        X = self.train_processed[0][self.__columnsFeature()]
        np.log1p(y).rename('log_' + y.name)
        self.model.fit(X, y)
        return self

    def predict(self):
        raw_output = self.model.predict(self.train_processed)
        return self.__postprocess(raw_output)

    def evaluation(self):
        print("Démarrage de l'evalution")

        if not self.test_processed:
            raise ValueError("Le test n'a pas été traité. Veuillez d'abord traiter les données de test.")

        X_train = self.train_processed[0]
        X_test = self.test_processed[0]

        y_train = self.train_processed[1]
        y_test = self.test_processed[1]

        columnsFeature = self.__columnsFeature()

        y_pred_train = self.model.predict(X_train[columnsFeature])
        y_pred_test = self.model.predict(X_test[columnsFeature])

        print("Train RMSLE = %.4f" % root_mean_squared_error(y_train, y_pred_train))
        print("Test RMSLE = %.4f" % root_mean_squared_error(y_test, y_pred_test))
        print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))