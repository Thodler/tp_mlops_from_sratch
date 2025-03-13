from src.data_processing import data_processing
from src.train_model import pipeline_processing, evaluation, serialisation


def main():
    set_train, set_test = data_processing()

    model = pipeline_processing(set_train)
    evaluation(model, set_train, set_test)

    serialisation(model)


if __name__ == "__main__":
    main()
