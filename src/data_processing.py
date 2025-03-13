import os
from pathlib import Path

import pandas as pd
import requests
import zipfile
import io

from sklearn.model_selection import train_test_split

from utils.config_loader import load_config
from utils.sqlite_service import SQLiteService

config = load_config()

my_db = SQLiteService()  # Connection a la BDD

URL = config['path']['url']
PATH_CSV = config['path']['extract_csv']
FILENAME_CSV = config['value']['filename_data']

def download_and_extract_zip():
    # Télécharger le fichier
    response = requests.get(URL)

    # Extraire le fichier ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(PATH_CSV)

    print(f"Fichier extrait dans le répertoire : {PATH_CSV}")

def prepare_database():
    path = Path(PATH_CSV) / FILENAME_CSV
    df = pd.read_csv(path)
    os.remove(path)

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train.to_sql('train', my_db.conn, if_exists='replace', index=False)
    X_test.to_sql('test', my_db.conn, if_exists='replace', index=False)

    print('Base de donnée crée.')

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
    download_and_extract_zip()
    prepare_database()
    result = clean_data()

    my_db.close()

    return result

if __name__ == "__main__":
    data_processing()
