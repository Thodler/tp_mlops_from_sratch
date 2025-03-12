import requests
import zipfile
import io

from utils.config_loader import load_config

config = load_config()

URL = config['path']['url']
PATH_CSV = config['path']['extract_csv']

def download_and_extract_zip():
    # Télécharger le fichier
    response = requests.get(URL)
    response.raise_for_status()  # Vérifier que le téléchargement s'est bien passé

    # Extraire le fichier ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(PATH_CSV)

    print(f"Fichier extrait dans le répertoire : {PATH_CSV}")


def data_processing():

    download_and_extract_zip()


if __name__ == "__main__":
    download_and_extract_zip()