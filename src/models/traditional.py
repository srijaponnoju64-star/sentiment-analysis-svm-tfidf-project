"""Traditional ML models: Naive Bayes, Logistic Regression, Linear SVM."""
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import config


def build_naive_bayes():
    return MultinomialNB(**config.NB_PARAMS)


def build_logistic_regression():
    return LogisticRegression(class_weight="balanced", **config.LOGREG_PARAMS)


def build_svm():
    # LinearSVC scales to tens of thousands of rows; SVC(kernel="linear")
    # does not — it becomes impractically slow past a few thousand samples.
    return LinearSVC(class_weight="balanced", random_state=config.RANDOM_STATE, max_iter=5000)


MODEL_BUILDERS = {
    "naive_bayes": build_naive_bayes,
    "logistic_regression": build_logistic_regression,
    "svm": build_svm,
}


def train_model(model_name, X_train, y_train):
    model = MODEL_BUILDERS[model_name]()
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)