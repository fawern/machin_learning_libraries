import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import regression

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


data ={
    'house_size': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000],
    'house_price':  [150, 225, 280, 330, 400, 460, 510, 575, 620, 680, 725, 770, 820]
}

data = pd.DataFrame(data)

y = data["house_price"]
x = data.drop(columns="house_price")


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

linear_model = regression.Regression(
    model=LinearRegression(), train_data=X_train, test_data=X_test,train_target=y_train, test_target=y_test
)

svr_model = regression.Regression(
    model=SVR(), train_data=X_train, test_data=X_test, train_target=y_train, test_target=y_test
)

print('Logistic Regression accuracy', linear_model.fit_predict())
print('SVC accuracy', svr_model.fit_predict())


print('---------------------------------------------------------')


(
    models_output_list,
    highest_r2_output,
    highest_mse_output,
    highest_mae_output,
) = regression.Regression.regression_models(X_train, X_test, y_train, y_test)


print("Regression Models and Evaluation Results:")

print("\nModel with the Highest R-squared (r2_score):")
print(highest_r2_output)

print("\nModel with the Lowest Mean Squared Error (MSE):")
print(highest_mse_output)

print("\nModel with the Lowest Mean Absolute Error (MAE):")
print(highest_mae_output)

for model_output in models_output_list:
    model_name = model_output[0]
    results = model_output[1]
    print(f"{model_name}: {results}")