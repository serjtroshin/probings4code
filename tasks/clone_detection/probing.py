# import random
# import warnings

# import numpy as np
# from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
# from sklearn.metrics import (accuracy_score, f1_score, precision_score,
#                              recall_score)
# from sklearn.model_selection import train_test_split

# from src.utils import Saver, Setup


# def load_data(dataset="code_clone", model="clone_detection", embeddings="s0", base_path="."):
#     path = str(Setup.get_raw_path(dataset, model, embeddings, base_path=base_path))
#     print(path)
#     saver = Saver(path)
#     data = saver.load()
#     return data

# def _make_dataset(data, model_layer=1):
#   X = []
#   y = []
#   for sample in data:
#     X.append(
#         sample["outputs"][model_layer].numpy()
#     )
#     y.append(sample["target"])
#   return np.array(X), np.array(y).reshape(-1, 1)


# def score(true, pred):
#     return {
#       "acc": accuracy_score(true, pred),
#       "precision": precision_score(true, pred),
#       "recall": recall_score(true, pred),
#       "f1": f1_score(true, pred)
#     }


# def train_linear_model(X_train, X_test, y_train, y_test):
#     model = RidgeClassifierCV()
#     model.fit(X_train, y_train.ravel())
#     return {"train": score(y_train, model.predict(X_train).reshape(-1, 1)),
#         "test": score(y_test, model.predict(X_test).reshape(-1, 1))}


# def train_for_layers(data, max_layers=7):
#   # train for sent1 <s> sent2 data
#   layer2qual = {}
#   for layer in range(0, max_layers):
#     X, y = make_dataset(data, model_layer=layer)
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#     qual = train_linear_model(X_train, X_test, y_train, y_test)
#     layer2qual[layer] = qual
#   return layer2qual


# from sklearn.manifold import TSNE


# def make_tsne(data, max_layers):
#     result = []

#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         # Your problematic instruction(s) here


#         for LAYER in range(0, max_layers):

#             random.seed(0)
#             np.random.seed(0)
#             X, y = make_dataset(data, model_layer=LAYER)

#             X_embeddings, labels = X, y

#             random.seed(0)

#             tsne = TSNE(metric="cosine")
#             latents = tsne.fit_transform(X_embeddings)
#             result.append((latents, labels, LAYER))
#     return result
