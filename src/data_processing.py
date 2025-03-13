import pandas as pd

from utils.config_loader import load_config
from utils.sqlite_service import SQLiteService

config = load_config()

my_db = SQLiteService()  # Connection a la BDD

def clean_data():
    # Requête en BDD
    row_train, columns =  my_db.get_df('train')
    row_test, columns =  my_db.get_df('test')

    df_train = pd.DataFrame(row_train, columns=columns)
    df_test = pd.DataFrame(row_test, columns=columns)

    X_train, y_train = preprocessing(df_train)
    X_test, y_test = preprocessing(df_test)

    return (X_train, y_train), (X_test, y_test)

def preprocessing(df):
    df.drop(columns=['id'], inplace=True)
    df.drop(columns=['dropoff_datetime'], inplace=True)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    X = df.drop(columns=['trip_duration'])
    y = df['trip_duration']

    X['pickup_date'] = X['pickup_datetime'].dt.date

    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]

    X['weekday'] = X['pickup_datetime'].dt.weekday
    X['month'] = X['pickup_datetime'].dt.month
    X['hour'] = X['pickup_datetime'].dt.hour
    X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)

    return X, y


def data_processing():
    result = clean_data()

    my_db.close()

    return result

if __name__ == "__main__":
    data_processing()
