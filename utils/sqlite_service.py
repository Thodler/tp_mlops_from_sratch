import os.path
import sqlite3

from utils.config_loader import load_config


class SQLiteService:

    def __init__(self):
        self.__init_config()
        self.conn = self.connect()

    def connect(self):
        return sqlite3.connect(self.__path_database())

    def close(self):
        self.conn.close()

    def __init_config(self):
        config = load_config()
        self.__PATH_DB = config['path']['extract_db']
        self.__DATABASE_NAME = config['value']['database_name']

    def __path_database(self):
        return os.path.join(self.__PATH_DB, self.__DATABASE_NAME)

    def get_df(self, table):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM " + table)

        return cursor.fetchall(), [description[0] for description in cursor.description]
