import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import classification as cls


data = pd.read_csv("your_data.csv")

y = data["output"]
x = data.drop(columns="output")


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

(
    models_output_list,
    highest_acs_output,
    highest_f1s_output,
    highest_prs_output,
    highest_rcs_output,
    highest_rocs_output,
) = cls.classification.classification_models(x_train, x_test, y_train, y_test)


print("Classification Models and Evaluation Results:")
for model_output in models_output_list:
    model_name = model_output[0]
    evaluation_results = model_output[1]
    print(f"{model_name}: {evaluation_results}")

print("\nModel with the Highest Accuracy:")
print(highest_acs_output)

print("\nModel with the Highest F1 Score:")
print(highest_f1s_output)

print("\nModel with the Highest Precision:")
print(highest_prs_output)

print("\nModel with the Highest Recall:")
print(highest_rcs_output)

print("\nModel with the Highest ROC-AUC Score:")
print(highest_rocs_output)
