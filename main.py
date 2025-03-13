import sys

from src.data_repository import data_repository
from src.download_data import download_data
from src.models.TaxiModel import TaxiModel
from src.train_model import pipeline_processing, serialisation
from utils.config_loader import load_config
from utils.feature import column_feature

config = load_config()

TARGET = config['value']['target']

def main(create_db = False):

    if create_db:
        download_data()

    set_train, set_test = data_repository()

    pipeline = pipeline_processing()

    taxi_model = TaxiModel(pipeline, column_feature())
    taxi_model.fit(set_train, set_train[TARGET], set_test, set_test[TARGET])

    serialisation(taxi_model)


if __name__ == "__main__":
    create_db = False

    if len(sys.argv) > 1 :
        create_db = sys.argv[1] == '--create_db'

    main(create_db)
