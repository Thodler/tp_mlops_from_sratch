import sys

from src.data_processing import data_processing
from src.get_data import get_data
from src.train_model import pipeline_processing, evaluation, serialisation


def main(create_db = False):

    if create_db:
        get_data()

    set_train, set_test = data_processing()

    model = pipeline_processing(set_train)
    evaluation(model, set_train, set_test)

    serialisation(model)


if __name__ == "__main__":
    create_db = False

    if len(sys.argv) > 1 :
        create_db = sys.argv[1] == '--create_db'

    main(create_db)
