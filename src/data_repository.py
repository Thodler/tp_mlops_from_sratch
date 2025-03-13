import pandas as pd

from utils.config_loader import load_config
from utils.sqlite_service import SQLiteService

config = load_config()

my_db = SQLiteService()  # Connection a la BDD

def get_train_and_test():
    # Requête en BDD
    row_train, columns =  my_db.get_df('train')
    row_test, columns =  my_db.get_df('test')

    df_train = pd.DataFrame(row_train, columns=columns)
    df_test = pd.DataFrame(row_test, columns=columns)

    return df_train, df_test

def data_repository():
    result = get_train_and_test()

    my_db.close()

    return result

if __name__ == "__main__":
    data_repository()
