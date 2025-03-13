import sys

from src.data_repository import data_repository
from src.download_data import download_data
from src.models.TaxiModel import TaxiModel
from src.train_model import pipeline_processing, serialisation


def main(create_db = False):

    if create_db:
        download_data()

    set_train, set_test = data_repository()

    pipeline = pipeline_processing()

    taxi_model = TaxiModel(pipeline, set_train, set_test)

    taxi_model.fit()
    taxi_model.evaluation()

    serialisation(taxi_model)


if __name__ == "__main__":
    create_db = False

    if len(sys.argv) > 1 :
        create_db = sys.argv[1] == '--create_db'

    main(create_db)
