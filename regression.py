import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
import xgboost as xgbr
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Regression:

    def __init__(self, model, train_data, test_data, train_target, test_target):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = train_target
        self.test_target = test_target

    def fit_predict(self):
        self.model.fit(self.train_data, self.train_target)
        return self.model.score(self.test_data, self.test_target)


def regression_models(x_train, x_test, y_train, y_test):
    models = [
        LinearRegression(),
        Ridge(),
        Lasso(alpha=0.3),
        ElasticNet(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        xgbr.XGBRegressor(),
        xgbr.XGBRFRegressor(),
        lgbm.LGBMRegressor(num_leaves=6),
        AdaBoostRegressor(),
        BayesianRidge(),
        MLPRegressor(),
    ]

    models_output_list = []

    for model in models:
        reg = Regression(model, x_train, x_test, y_train, y_test)
        reg_score = reg.fit_predict()
        y_pred = model.predict(x_test)

        models_output_list.append({
            'Model': type(model).__name__,
            'r2_score': r2_score(y_test, y_pred),
            'mean_squared_error': mean_squared_error(y_test, y_pred),
            'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        })

    models_output_df = pd.DataFrame(models_output_list)

    highest_r2_output = models_output_df.loc[models_output_df['r2_score'].idxmax()]
    lowest_mse_output = models_output_df.loc[models_output_df['mean_squared_error'].idxmin()]
    lowest_mae_output = models_output_df.loc[models_output_df['mean_absolute_error'].idxmin()]

    return models_output_df, highest_r2_output, lowest_mse_output, lowest_mae_output
