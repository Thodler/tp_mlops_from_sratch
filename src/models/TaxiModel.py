import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

class TaxiModel:

    __TARGET_NAME = 'trip_duration'

    def __init__(self, model, column_feature):
        self.model = model
        self.__columnsFeature = column_feature

    def __preprocess(self, X):
        print(X.head())

        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
        X['pickup_date'] = X['pickup_datetime'].dt.date

        df_abnormal_dates = X.groupby('pickup_date').size()
        abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]

        X['weekday'] = X['pickup_datetime'].dt.weekday
        X['month'] = X['pickup_datetime'].dt.month
        X['hour'] = X['pickup_datetime'].dt.hour
        X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)

        return X[self.__columnsFeature]

    def __postprocess(self, raw_output):
        # your postprocessing logic: inverse transformation, etc.
        return np.expm1(np.round(raw_output))


    def fit(self, X, y, X_test, y_test):
        X = self.__preprocess(X)
        y_log = np.log1p(y).rename('log_' + y.name)
        self.model.fit(X, y_log)

        self.__evaluation(X, y, X_test, y_test)
        return self

    def predict(self, X):
        X = self.__preprocess(X)
        raw_output = self.model.predict(X)
        return self.__postprocess(raw_output)

    def __evaluation(self, set_train, y_train, set_test, y_test):
        print("Démarrage de l'evalution")

        X_test = self.__preprocess(set_test)

        y_pred_train = self.model.predict(set_train[self.__columnsFeature])
        y_pred_test = self.model.predict(X_test[self.__columnsFeature])

        print("Train RMSLE = %.4f" % root_mean_squared_error(y_train, y_pred_train))
        print("Test RMSLE = %.4f" % root_mean_squared_error(y_test, y_pred_test))
        print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))