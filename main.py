from src.data_processing import data_processing

def main():
    set_train, set_test = data_processing()

    X_train, y_train = set_train


if __name__ == "__main__":
    main()