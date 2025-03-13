from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_features = ['abnormal_period', 'hour']
cat_features = ['weekday', 'month']

train_features = num_features + cat_features

def pipeline_processing(df_train):
    X_train, y_train = df_train

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    return pipeline.fit(X_train[train_features], y_train)

def evaluation(model, df_train, df_test):

    X_train, y_train = df_train
    X_test, y_test = df_test

    y_pred_train = model.predict(X_train[train_features])
    y_pred_test = model.predict(X_test[train_features])

    print("Train RMSLE = %.4f" % root_mean_squared_error(y_train, y_pred_train))
    print("Test RMSLE = %.4f" % root_mean_squared_error(y_test, y_pred_test))
    print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))