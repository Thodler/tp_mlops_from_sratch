from src.data_processing import data_processing
from src.train_model import pipeline_processing, evaluation


def main():
    set_train, set_test = data_processing()

    models = pipeline_processing(set_train)
    evaluation(models, set_train, set_test)


if __name__ == "__main__":
    main()
