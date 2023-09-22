import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import classification as cls_

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data = {
    'Gelir': [50000, 60000, 30000, 80000, 40000, 55000, 62000, 48000, 70000, 75000, 42000, 68000, 55000, 62000, 40000, 58000, 67000, 43000, 71000, 55000, 60000, 40000, 65000, 72000, 45000, 59000, 68000, 36000, 69000, 70000],
    'Kredi Puanı': [700, 750, 600, 800, 650, 720, 740, 710, 680, 720, 670, 750, 680, 730, 640, 760, 690, 710, 780, 720, 730, 650, 740, 760, 690, 720, 750, 670, 770, 780],
    'Borç Oranı ': [0.2, 0.3, 0.5, 0.1, 0.4, 0.25, 0.15, 0.3, 0.4, 0.2, 0.35, 0.15, 0.3, 0.25, 0.45, 0.2, 0.35, 0.4, 0.2, 0.25, 0.3, 0.4, 0.35, 0.15, 0.2, 0.25, 0.3, 0.45, 0.15, 0.2],
    'Kredi Durumu': ['Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Onaylandı', 'Onaylandı', 'Onaylandı', 'Reddedildi', 'Reddedildi']
}


df = pd.DataFrame(data)


X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

y = y.map({'Onaylandı': 1, 'Reddedildi': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

logreg_model = cls_.Classification(
    model=LogisticRegression(), 
    train_data=X_train, test_data=X_test, 
    train_target=y_train, test_target=y_test
)

svc_model = cls_.Classification(
    model=SVC(),
    train_data=X_train, test_data=X_test,
    train_target=y_train, test_target=y_test
)
rf_model = cls_.Classification(
    model=RandomForestClassifier(),
    train_data=X_train, test_data=X_test,
    train_target=y_train, test_target=y_test
)

print('Logistic Regression accuracy', logreg_model.fit_predict())
print('SVC accuracy', svc_model.fit_predict())
print('Random Forest accuracy', rf_model.fit_predict())

print('---------------------------------------------------------')

(
    models_output_list,
    highest_acs_output,
    highest_f1s_output,
    highest_prs_output,
    highest_rcs_output,
    highest_rocs_output,
) = cls_.Classification.classification_models(X_train, X_test, y_train, y_test)


print("Classification Models and Evaluation Results:")

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

for model_output in models_output_list:
    model_name = model_output[0]
    evaluation_results = model_output[1]
    print(f"{model_name}: {evaluation_results}")