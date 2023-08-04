import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgbr
import xgboost
import lightgbm as lgbm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings("ignore")


class regression:
    def regression_models(x_train, x_test, y_train, y_test):
        models_output_list = []

        #! linear regression
        model_lr = LinearRegression()

        model_lr.fit(x_train, y_train)
        y_pred_lr = model_lr.predict(x_test)
        models_output_list.append(
            [
                "LinearRegression",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_lr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_lr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_lr),
                ],
            ]
        )

        #! Ridge
        model_ridge = Ridge()

        model_ridge.fit(x_train, y_train)
        y_pred_ridge = model_ridge.predict(x_test)

        models_output_list.append(
            [
                "Ridge",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_ridge),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_ridge),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_ridge),
                ],
            ]
        )

        #! Lasso
        model_lasso = Lasso(alpha=0.3)

        model_lasso.fit(x_train, y_train)
        y_pred_lasso = model_lasso.predict(x_test)

        models_output_list.append(
            [
                "Lasso",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_lasso),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_lasso),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_lasso),
                ],
            ]
        )

        #! ElasticNet
        model_elasticnet = ElasticNet()

        model_elasticnet.fit(x_train, y_train)
        y_pred_elasticnet = model_elasticnet.predict(x_test)
        models_output_list.append(
            [
                "ElasticNet",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_elasticnet),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_elasticnet),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_elasticnet),
                ],
            ]
        )

        #! KNN
        model_knn = KNeighborsRegressor()

        model_knn.fit(x_train, y_train)
        y_pred_knn = model_knn.predict(x_test)
        models_output_list.append(
            [
                "KNeighborsRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_knn),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_knn),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_knn),
                ],
            ]
        )

        #! Decision Tree
        model_dt = DecisionTreeRegressor()

        model_dt.fit(x_train, y_train)
        y_pred_dt = model_dt.predict(x_test)
        models_output_list.append(
            [
                "DecisionTreeRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_dt),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_dt),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_dt),
                ],
            ]
        )

        #! SVM
        model_svr = SVR()

        model_svr.fit(x_train, y_train)
        y_pred_svr = model_svr.predict(x_test)
        models_output_list.append(
            [
                "SVR",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_svr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_svr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_svr),
                ],
            ]
        )

        #! RandomForest
        model_rfr = RandomForestRegressor()

        model_rfr.fit(x_train, y_train)
        y_pred_rfr = model_rfr.predict(x_test)
        models_output_list.append(
            [
                "RandomForestRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_rfr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_rfr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_rfr),
                ],
            ]
        )

        #! GradientBoosting
        model_gbr = GradientBoostingRegressor()
        model_gbr.fit(x_train, y_train)
        y_pred_gbr = model_gbr.predict(x_test)
        models_output_list.append(
            [
                "GradientBoostingRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_gbr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_gbr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_gbr),
                ],
            ]
        )

        #! XGBRegressor
        model_xgb = xgbr.XGBRegressor()

        model_xgb.fit(x_train, y_train)
        y_pred_xgb = model_xgb.predict(x_test)
        models_output_list.append(
            [
                "XGBRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_xgb),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_xgb),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_xgb),
                ],
            ]
        )

        #! XGBRFRegressor
        model_xgbrf = xgbr.XGBRFRegressor()

        model_xgbrf.fit(x_train, y_train)
        y_pred_xgbrf = model_xgbrf.predict(x_test)
        models_output_list.append(
            [
                "XGBRFRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_xgbrf),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_xgbrf),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_xgbrf),
                ],
            ]
        )

        #! LightGBM
        model_lgbm = lgbm.LGBMRegressor(num_leaves=6)

        model_lgbm.fit(x_train, y_train)
        y_pred_lgbm = model_lgbm.predict(x_test)
        models_output_list.append(
            [
                "LGBMRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_lgbm),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_lgbm),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_lgbm),
                ],
            ]
        )

        #! AdaBoostRegressor
        model_abr = AdaBoostRegressor()

        model_abr.fit(x_train, y_train)
        y_pred_abr = model_abr.predict(x_test)
        models_output_list.append(
            [
                "AdaBoostRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_abr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_abr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_abr),
                ],
            ]
        )

        #! BayesianRidge
        model_byr = BayesianRidge()

        model_byr.fit(x_train, y_train)
        y_pred_byr = model_byr.predict(x_test)
        models_output_list.append(
            [
                "BayesianRidge",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_byr),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_byr),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_byr),
                ],
            ]
        )

        #! MLPRegressor
        model_mlpc = MLPRegressor()

        model_mlpc.fit(x_train, y_train)
        y_pred_mlpc = model_mlpc.predict(x_test)
        models_output_list.append(
            [
                "MLPRegressor",
                [
                    "r2_score:",
                    r2_score(y_test, y_pred_mlpc),
                    "mean_squared_error:",
                    mean_squared_error(y_test, y_pred_mlpc),
                    "mean_absolute_error:",
                    mean_absolute_error(y_test, y_pred_mlpc),
                ],
            ]
        )

        highest_r2_output = [None, 0]
        highest_mse_output = [None, models_output_list[0][1][3]]
        highest_mae_output = [None, models_output_list[0][1][5]]
        for a in range(12):
            for b in range(6):
                if b == 1:
                    output_r2_value = models_output_list[a][1][1]
                    if output_r2_value > highest_r2_output[1]:
                        highest_r2_output[0] = models_output_list[a][0]
                        highest_r2_output[1] = output_r2_value
                if b == 3:
                    output_mse_value = models_output_list[a][1][3]
                    if output_mse_value < highest_mse_output[1]:
                        highest_mse_output[0] = models_output_list[a][0]
                        highest_mse_output[1] = output_mse_value
                if b == 5:
                    output_mae_value = models_output_list[a][1][5]
                    if output_mae_value < highest_mae_output[1]:
                        highest_mae_output[0] = models_output_list[a][0]
                        highest_mae_output[1] = output_mae_value
        return (
            models_output_list,
            highest_r2_output,
            highest_mse_output,
            highest_mae_output,
        )
