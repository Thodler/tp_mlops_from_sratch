import os.path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config_loader import load_config
from utils.feature import cat_features, num_features

config = load_config()

def pipeline_processing():
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features()),
        ('scaling', StandardScaler(), num_features())]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    return pipeline

def serialisation(model):
    print("Enregistrement du model")
    joblib.dump(model, os.path.join(config['path']['model'], config['value']['model_name']))
    print("Enregistrement terminé avec succes.")
