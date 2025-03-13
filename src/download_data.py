import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config
from utils.sqlite_service import SQLiteService

config = load_config()

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

    my_db = SQLiteService()  # Connection a la BDD

    X_train.to_sql('train', my_db.conn, if_exists='replace', index=False)
    X_test.to_sql('test', my_db.conn, if_exists='replace', index=False)

    my_db.close()
    print('Base de donnée crée.')

def download_data():
    download_and_extract_zip()
    prepare_database()

if __name__ == "__main__":
    download_data()
