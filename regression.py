import pandas as pd
from sklearn.model_selection import train_test_split 

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

    def __init__(self, model, data, y_col, test_size):
        self.model = model
        self.data = data
        self.y_col = y_col 
        self.test_size = test_size

        X = self.data.drop(columns=[self.y_col])
        y = self.data[y_col]

        self.train_data , self.test_data, self.train_target, self.test_target = train_test_split(X, y, test_size=self.test_size, random_state=13)

    def fit_predict(self):
        self.model.fit(self.train_data, self.train_target)
        self.y_pred = self.model.predict(self.test_data)
        
        return self.y_pred, self.model.score(self.test_data, self.test_target)


def regression_models(data, y_col, test_size):
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
        reg = Regression(model, data, y_col, test_size)
        predictions, _ = reg.fit_predict()

        rs = r2_score(reg.test_target, predictions)
        mse = mean_squared_error(reg.test_target, predictions)
        mae = mean_absolute_error(reg.test_target, predictions)

        models_output_list.append({
            'Model': type(model).__name__,
            'r2_score': rs,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
        })

    models_output_df = pd.DataFrame(models_output_list)

    highest_r2_output = models_output_df.loc[models_output_df['r2_score'].idxmax()]
    lowest_mse_output = models_output_df.loc[models_output_df['mean_squared_error'].idxmin()]
    lowest_mae_output = models_output_df.loc[models_output_df['mean_absolute_error'].idxmin()]

    return models_output_df, highest_r2_output, lowest_mse_output, lowest_mae_output
