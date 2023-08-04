import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import regression


data = pd.read_csv("your_data.csv")

y = data["output"]
x = data.drop(columns="output")


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


(
    models_output_list,
    highest_r2_output,
    highest_mse_output,
    highest_mae_output,
) = regression.regression.regression_models(x_train, x_test, y_train, y_test)


print("Regression Models and Evaluation Results:")
for model_output in models_output_list:
    model_name = model_output[0]
    results = model_output[1]
    print(f"{model_name}: {results}")

print("\nModel with the Highest R-squared (r2_score):")
print(highest_r2_output)

print("\nModel with the Lowest Mean Squared Error (MSE):")
print(highest_mse_output)

print("\nModel with the Lowest Mean Absolute Error (MAE):")
print(highest_mae_output)
