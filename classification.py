import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")


class Classification:

    def __init__(self, model, train_data, test_data, train_target, test_target):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = train_target
        self.test_target = test_target

    def fit_predict(self):
        self.model.fit(self.train_data, self.train_target)
        self.y_pred = self.model.predict(self.test_data)

        return accuracy_score(self.test_target, self.y_pred)

    def classification_models(x_train, x_test, y_train, y_test):
        models_output_list = []

        #! logistic
        model_lr = LogisticRegression(max_iter=990)

        model_lr.fit(x_train, y_train)
        y_pred_lr = model_lr.predict(x_test)
        models_output_list.append(
            [
                "LogisticRegression",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_lr),
                    "precision_score",
                    precision_score(y_test, y_pred_lr),
                    "f1_score",
                    f1_score(y_test, y_pred_lr),
                    "recall_score",
                    recall_score(y_test, y_pred_lr),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_lr),
                ],
            ]
        )

        #! KNeighborsClassifier
        model_knnc = KNeighborsClassifier()

        model_knnc.fit(x_train, y_train)
        y_pred_knnc = model_knnc.predict(x_test)
        models_output_list.append(
            [
                "KNeighborsClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_knnc),
                    "precision_score",
                    precision_score(y_test, y_pred_knnc),
                    "f1_score",
                    f1_score(y_test, y_pred_knnc),
                    "recall_score",
                    recall_score(y_test, y_pred_knnc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_knnc),
                ],
            ]
        )
        #! DecisionTreeClassifier
        model_dtc = DecisionTreeClassifier()

        model_dtc.fit(x_train, y_train)
        y_pred_dtc = model_dtc.predict(x_test)
        models_output_list.append(
            [
                "DecisionTreeClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_dtc),
                    "precision_score",
                    precision_score(y_test, y_pred_dtc),
                    "f1_score",
                    f1_score(y_test, y_pred_dtc),
                    "recall_score",
                    recall_score(y_test, y_pred_dtc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_dtc),
                ],
            ]
        )
        #! SVC
        model_svc = SVC()

        model_svc.fit(x_train, y_train)
        y_pred_svc = model_svc.predict(x_test)
        models_output_list.append(
            [
                "SVC",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_svc),
                    "precision_score",
                    precision_score(y_test, y_pred_svc),
                    "f1_score",
                    f1_score(y_test, y_pred_svc),
                    "recall_score",
                    recall_score(y_test, y_pred_svc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_svc),
                ],
            ]
        )
        #! RandomForestClassifier
        model_rfc = RandomForestClassifier(random_state=42)

        model_rfc.fit(x_train, y_train)
        y_pred_rfc = model_rfc.predict(x_test)
        models_output_list.append(
            [
                "RandomForestClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_rfc),
                    "precision_score",
                    precision_score(y_test, y_pred_rfc),
                    "f1_score",
                    f1_score(y_test, y_pred_rfc),
                    "recall_score",
                    recall_score(y_test, y_pred_rfc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_rfc),
                ],
            ]
        )
        #! GradientBoostingClassifier
        model_gbc = GradientBoostingClassifier()

        model_gbc.fit(x_train, y_train)
        y_pred_gbc = model_gbc.predict(x_test)
        models_output_list.append(
            [
                "GradientBoostingClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_gbc),
                    "precision_score",
                    precision_score(y_test, y_pred_gbc),
                    "f1_score",
                    f1_score(y_test, y_pred_gbc),
                    "recall_score",
                    recall_score(y_test, y_pred_gbc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_gbc),
                ],
            ]
        )
        #! XGBClassifier
        model_xgbc = XGBClassifier()

        model_xgbc.fit(x_train, y_train)
        y_pred_xgbc = model_xgbc.predict(x_test)
        models_output_list.append(
            [
                "XGBClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_xgbc),
                    "precision_score",
                    precision_score(y_test, y_pred_xgbc),
                    "f1_score",
                    f1_score(y_test, y_pred_xgbc),
                    "recall_score",
                    recall_score(y_test, y_pred_xgbc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_xgbc),
                ],
            ]
        )
        #! XGBRFClassifier
        model_xgbrfc = XGBRFClassifier()

        model_xgbrfc.fit(x_train, y_train)
        y_pred_xgbrfc = model_xgbrfc.predict(x_test)
        models_output_list.append(
            [
                "XGBRFClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_xgbrfc),
                    "precision_score",
                    precision_score(y_test, y_pred_xgbrfc),
                    "f1_score",
                    f1_score(y_test, y_pred_xgbrfc),
                    "recall_score",
                    recall_score(y_test, y_pred_xgbrfc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_xgbrfc),
                ],
            ]
        )
        #! LGBMClassifier
        model_lgbmc = LGBMClassifier()

        model_lgbmc.fit(x_train, y_train)
        y_pred_lgbmc = model_lgbmc.predict(x_test)
        models_output_list.append(
            [
                "LGBMClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_lgbmc),
                    "precision_score",
                    precision_score(y_test, y_pred_lgbmc),
                    "f1_score",
                    f1_score(y_test, y_pred_lgbmc),
                    "recall_score",
                    recall_score(y_test, y_pred_lgbmc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_lgbmc),
                ],
            ]
        )
        #! CatBoostClassifier
        model_cbc = CatBoostClassifier(verbose=False)

        model_cbc.fit(x_train, y_train)
        y_pred_cbc = model_cbc.predict(x_test)
        models_output_list.append(
            [
                "CatBoostClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_cbc),
                    "precision_score",
                    precision_score(y_test, y_pred_cbc),
                    "f1_score",
                    f1_score(y_test, y_pred_cbc),
                    "recall_score",
                    recall_score(y_test, y_pred_cbc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_cbc),
                ],
            ]
        )

        #! GaussianNB
        model_gnb = GaussianNB()

        model_gnb.fit(x_train, y_train)
        y_pred_gnb = model_gnb.predict(x_test)
        models_output_list.append(
            [
                "GaussianNB",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_gnb),
                    "precision_score",
                    precision_score(y_test, y_pred_gnb),
                    "f1_score",
                    f1_score(y_test, y_pred_gnb),
                    "recall_score",
                    recall_score(y_test, y_pred_gnb),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_gnb),
                ],
            ]
        )

        #! MLPClassifier
        model_mlpc = MLPClassifier()

        model_mlpc.fit(x_train, y_train)
        y_pred_mlpc = model_mlpc.predict(x_test)
        models_output_list.append(
            [
                "MLPClassifier",
                [
                    "accuracy_score",
                    accuracy_score(y_test, y_pred_mlpc),
                    "precision_score",
                    precision_score(y_test, y_pred_mlpc),
                    "f1_score",
                    f1_score(y_test, y_pred_mlpc),
                    "recall_score",
                    recall_score(y_test, y_pred_mlpc),
                    "roc_auc_score",
                    roc_auc_score(y_test, y_pred_mlpc),
                ],
            ]
        )

        highest_acs_output = [None, 0]
        highest_prs_output = [None, 0]
        highest_f1s_output = [None, 0]
        highest_rcs_output = [None, 0]
        highest_rocs_output = [None, 0]
        for a in range(10):
            for b in range(10):
                if b == 1:
                    output_acs_value = models_output_list[a][1][1]
                    if output_acs_value > highest_acs_output[1]:
                        highest_acs_output[0] = models_output_list[a][0]
                        highest_acs_output[1] = output_acs_value

                if b == 3:
                    output_prs_value = models_output_list[a][1][3]
                    if output_prs_value > highest_prs_output[1]:
                        highest_prs_output[0] = models_output_list[a][0]
                        highest_prs_output[1] = output_prs_value
                if b == 5:
                    output_f1s_value = models_output_list[a][1][5]
                    if output_f1s_value > highest_f1s_output[1]:
                        highest_f1s_output[0] = models_output_list[a][0]
                        highest_f1s_output[1] = output_f1s_value
                if b == 7:
                    output_rcs_value = models_output_list[a][1][7]
                    if output_rcs_value > highest_rcs_output[1]:
                        highest_rcs_output[0] = models_output_list[a][0]
                        highest_rcs_output[1] = output_rcs_value
                if b == 9:
                    output_rocs_value = models_output_list[a][1][9]
                    if output_rocs_value > highest_rocs_output[1]:
                        highest_rocs_output[0] = models_output_list[a][0]
                        highest_rocs_output[1] = output_rocs_value
        return (
            models_output_list,
            highest_acs_output,
            highest_f1s_output,
            highest_prs_output,
            highest_rcs_output,
            highest_rocs_output,
        )
