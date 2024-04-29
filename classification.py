import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
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

        return self.y_pred, accuracy_score(self.test_target, self.y_pred)


def classification_models(x_train, x_test, y_train, y_test, selected_models=None):
    models = [
        LogisticRegression(max_iter=990),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        SVC(),
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(),
        XGBClassifier(),
        XGBRFClassifier(),
        LGBMClassifier(),
        CatBoostClassifier(verbose=False),
        GaussianNB(),
        MLPClassifier(),
    ]

    if selected_models:
        models = selected_models
    else:
        models = models

    models_output_list = []

    for model in models:
        clf = Classification(model, x_train, x_test, y_train, y_test)
        predictions, _ = clf.fit_predict()

        acc_score = accuracy_score(y_test, predictions)
        prec_score = precision_score(y_test, predictions)
        f1 = f1_score(y_test, clf.y_pred)
        recall = recall_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        models_output_list.append({
            'Model': type(model).__name__,
            'Accuracy Score': acc_score,
            'Precision Score': prec_score,
            'F1 Score': f1,
            'Recall Score': recall,
            'Roc Auc Score': roc_auc
        })

    models_output_df = pd.DataFrame(models_output_list)

    highest_acs_output = models_output_df.loc[models_output_df['Accuracy Score'].idxmax()]
    highest_prs_output = models_output_df.loc[models_output_df['Precision Score'].idxmax()]
    highest_f1s_output = models_output_df.loc[models_output_df['F1 Score'].idxmax()]
    highest_rcs_output = models_output_df.loc[models_output_df['Recall Score'].idxmax()]
    highest_rocs_output = models_output_df.loc[models_output_df['Roc Auc Score'].idxmax()]

    return models_output_df, highest_acs_output, highest_prs_output, highest_f1s_output, highest_rcs_output, highest_rocs_output
