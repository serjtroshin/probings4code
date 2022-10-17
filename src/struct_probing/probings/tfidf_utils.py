from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeCV, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_tf_idf_model(model):
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(max_features=50000)),
            ("tfidf", TfidfTransformer()),
            ("clf", model),
        ]
    )
    return pipeline


def TfIdfClassifier():
    params = {"alpha": [0.0001, 0.001, 0.01, 0.1]}
    model = GridSearchCV(
        SGDClassifier(loss="log", verbose=0, tol=0.0001), param_grid=params, verbose=3
    )
    return get_tf_idf_model(model)


def TfIdfRegressor():
    model = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100))
    return get_tf_idf_model(model)
