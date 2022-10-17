# from numpy import inner
# import torch
# from torch import nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import StepLR
# from tqdm import tqdm
# import numpy as np
# import optuna
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error

# import logging

# class MLPBlock(nn.Module):
#     def __init__(self, ind, outd, dropout=0.0):
#         super().__init__()
#         self.layer = nn.Linear(ind, outd)
#         self.nonlinearity = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x):
#         x = self.layer(x)
#         x = self.nonlinearity(x)
#         x = self.dropout(x)
#         return x

# class MLP(nn.Module):
#     def __init__(self, input_dim, inner_dim, n_inner_layers, output_dim, n_epochs=100, dropout=0.0, cpu=False, step_size=10, lr=1e-4, gamma=0.1):
#         super().__init__()
#         # classifier: block -> block -> ... -> linear
#         # block: linear -> relu -> dropout
#         self.cpu = cpu
#         self.bs = 1024
#         self.n_epochs = n_epochs
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.layers = []
#         if n_inner_layers > 0:
#             self.layers.append(MLPBlock(input_dim, inner_dim, dropout))
#             for _ in range(n_inner_layers-1):
#                 self.layers.append(MLPBlock(inner_dim, inner_dim, dropout))
#             # print(inner_dim, output_dim)
#             self.layers.append(nn.Linear(inner_dim, output_dim))
#         else:
#             self.layers.append(nn.Linear(input_dim, output_dim))  # linear model

#         self.model = nn.Sequential(*self.layers)
#         if torch.cuda.is_available() and not self.cpu:
#             self.model = self.model.cuda()

#         self.optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=lr,
#             weight_decay=1e-1,
#         )
#         self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

#     def train_step(self, inputs, labels):
#         # return loss
#         raise NotImplemented

#     def train(self, hs, true_class_labels):
#         self.model.train()
#         device = "cpu"
#         if torch.cuda.is_available() and not self.cpu:
#             device = "cuda"
#         hs = torch.tensor(hs).to(device)
#         true_class_labels = torch.from_numpy(true_class_labels).to(device)

#         for n_epoch in range(self.n_epochs):
#             # with tqdm(range(0, hs.shape[0] - self.bs, self.bs)) as pbar:
#             #     for i in pbar:
#             loss_epoch = 0.0
#             for i in range(0, hs.shape[0] - self.bs, self.bs):
#                 inputs = hs[i: i+self.bs]
#                 labels = true_class_labels[i: i+self.bs]

#                 loss = self.train_step(inputs, labels).mean()
#                 loss_epoch = loss_epoch * 0.9 + loss.cpu().item() * 0.1
#                 # outputs = self.model(inputs)
#                 # print(outputs, labels)
#                 # loss = self.loss(outputs, labels).mean()
#                 # pbar.set_description(f"n_epoch> {n_epoch} loss train {loss.cpu().item():.4f}")
#                 loss.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#             self.scheduler.step()
#             logging.info(f"{n_epoch} epoch loss train {loss_epoch:.5f}")

#     def predict_step(self, inputs):
#         raise NotImplemented

#     def valid_metric(self, y_true, y_pred):
#         raise NotImplemented

#     def predict(self, hs):
#         self.model.eval()
#         device = "cpu"
#         if torch.cuda.is_available() and not self.cpu:
#             device = "cuda"
#         hs = torch.tensor(hs).to(device)
#         preds = []
#         with torch.no_grad():
#             with tqdm(range(0, hs.shape[0] - self.bs, self.bs)) as pbar:
#                 for i in pbar:
#                     inputs = hs[i: i+self.bs]
#                     outputs = self.predict_step(inputs)
# #                     outputs = self.model(inputs)
# #                     outputs = torch.argmax(outputs, dim=-1)
#                     preds.append(outputs)
#             if hs.shape[0] % self.bs != 0:
#                 inputs = hs[hs.shape[0] // self.bs * self.bs:]
#                 outputs = self.predict_step(inputs)
# #                 outputs = self.model(inputs)
# #                 outputs = torch.argmax(outputs, dim=-1)
#                 preds.append(outputs)
#         preds = torch.cat(preds, dim=0).cpu()
#         return preds

#     @staticmethod
#     def fit_optuna(Model, X_train, y_train, n_epochs=20, n_hidden=128, n_layers=0):
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)
#         def objective(trial):
#             # hyperparameters
#             dropout = trial.suggest_float("drop_p", 0.0, 1.0)
#             lr = trial.suggest_loguniform('lr', 1e-5, 10),
#             step_size = trial.suggest_int("step_size", 1, n_epochs, log=True)
#             # logging.info(str("n_hidden", n_hidden, "n_layers", n_layers, "drop", dropout, "lr", lr, "step_size", step_size))

#             model = Model(
#                 X_train.shape[-1],
#                 n_hidden,
#                 n_layers,
#                 output_dim=y_train.max() + 1,
#                 cpu=False,
#                 dropout=dropout,
#                 lr=float(lr[0]),
#                 step_size=step_size,
#                 n_epochs=n_epochs
#             )
#             logging.info(str(model))
#             model.train(np.array(X_train), np.array(y_train))
#             pred_val = model.predict(np.array(X_val))
#             score = model.valid_metric(y_val, pred_val)
#             return score
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=40)
#         logging.info(f"Return a dictionary of parameter name and parameter values: {study.best_params}")
#         best_params = study.best_params

#         return Model(
#                 X_train.shape[-1],
#                 n_hidden,
#                 n_layers,
#                 y_train.max() + 1,
#                 cpu=False,
#                 dropout=best_params["drop_p"],
#                 lr=best_params["lr"],
#                 step_size=best_params["step_size"],
#                 n_epochs=n_epochs
#             )


# class _MLPClassifier(MLP):
#     def __init__(self, input_dim, inner_dim, n_inner_layers, output_dim, n_epochs=100, dropout=0.0, cpu=False, step_size=10, lr=1e-4, gamma=0.1):
#         super().__init__(
#             input_dim=input_dim,
#             inner_dim=inner_dim,
#             n_inner_layers=n_inner_layers,
#             output_dim=output_dim,
#             n_epochs=n_epochs,
#             dropout=dropout,
#             cpu=cpu,
#             step_size=step_size,
#             lr=lr,
#             gamma=gamma,
#         )
#         self.loss = nn.CrossEntropyLoss()

#     def train_step(self, inputs, labels):
#         # return loss
#         outputs = self.model(inputs)
#         return self.loss(outputs, labels)

#     def predict_step(self, inputs):
#         outputs = self.model(inputs)
#         outputs = torch.argmax(outputs, dim=-1)
#         return outputs

#     def valid_metric(self, y_val, pred_val):
#         return accuracy_score(y_val, pred_val)


# class _MLPRegressor(MLP):
#     def __init__(self, input_dim, inner_dim, n_inner_layers, output_dim, n_epochs=100, dropout=0.0, cpu=False, step_size=10, lr=1e-4, gamma=0.1):
#         super().__init__(
#             input_dim=input_dim,
#             inner_dim=inner_dim,
#             n_inner_layers=n_inner_layers,
#             output_dim=1,  # for regression
#             n_epochs=n_epochs,
#             dropout=dropout,
#             cpu=cpu,
#             step_size=step_size,
#             lr=lr,
#             gamma=gamma,
#         )
#         self.loss = nn.MSELoss()

#     def train_step(self, inputs, labels):
#         # return loss
#         outputs = self.model(inputs).squeeze(-1)
#         return self.loss(outputs, labels.float())

#     def predict_step(self, inputs):
#         outputs = self.model(inputs).squeeze(-1)
#         return outputs

#     def valid_metric(self, y_val, pred_val):
#         return -mean_absolute_error(y_val, pred_val)


# class MLPSklearn:
#     def __init__(self, modelClass: MLP):
#         self.modelClass = modelClass

#     def fit(self, X_train, y_train):
#         self.best_model = MLP.fit_optuna(self.modelClass, X_train, y_train)
#         return self

#     def predict(self, X_train):
#         return self.best_model.predict(X_train)


# def MLPRegressor():
#     return MLPSklearn(_MLPRegressor)

# def MLPClassifier():
#     return MLPSklearn(_MLPClassifier)

import logging

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, mean_absolute_error

from .mlp_torch import MLP, Config, TorchClassifier, TorchRegressor

# from sklearn.neural_network import MLPClassifier as MLPClassifierSklearn
# from sklearn.neural_network import MLPRegressor as MLPRegressorSklearn
# # from skopt import BayesSearchCV

# def MLPRegressor():
#     class Model:
#         def __init__(self):
#             model = MLPRegressorSklearn(
#                 hidden_layer_sizes=(128, 128, 128), learning_rate="adaptive",
#                 max_iter=5000,
#                 early_stopping=True
#             )
#             cv = BayesSearchCV(model, {
#                 "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                 "learning_rate_init": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#             }, n_iter=15)
#             self.cv = cv

#         def fit(self, X_train, y_train):
#             _ = self.cv.fit(X_train, y_train)
#             logging.info(f"best_params: {self.cv.best_params_}")
#             return self.cv.best_estimator_

#         def predict(self, X_train):
#                 return self.cv.best_estimator_.predict(X_train)

#     return Model()

# def MLPClassifier():
#     class Model:
#         def __init__(self):
#             model = MLPClassifierSklearn(
#                 hidden_layer_sizes=(128, 128, 128), learning_rate="adaptive",
#                 max_iter=5000,
#                 early_stopping=True
#             )
#             cv = BayesSearchCV(model, {
#                 "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                 "learning_rate_init": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#             }, n_iter=15,)
#             self.cv = cv

#         def fit(self, X_train, y_train):
#             _ = self.cv.fit(X_train, y_train)
#             logging.info(f"best_params: {self.cv.best_params_}")
#             return self.cv.best_estimator_

#         def predict(self, X_train):
#             return self.cv.best_estimator_.predict(X_train)

#     return Model()




def TorchMLPRegressor():
    # 1. Define an objective function to be maximized.
    class Model:
        def __init__(self):
            pass

        def fit(self, X_train, y_train):
            n_features = X_train.shape[-1]

            def objective(trial: optuna.Trial):

                # 2. Suggest values of the hyperparameters using a trial object.
                lr = trial.suggest_float("lr", high=0.1, low=0.0001, log=True)
                weight_decay = trial.suggest_float(
                    "weight_decay", high=0.1, low=0.00001, log=True
                )

                config = Config(
                    verbose=False,
                    cuda_if_avail=True,
                    model=Config.Model(in_dim=n_features, n_hiddens=3, dropout=0),
                    training=Config.Training(
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=512,
                        num_updates=5000,
                        patience=10,
                    ),
                )
                model = TorchRegressor(config=config, model=MLP(args=config.model))
                train_losses, val_losses, X_val, y_val = model.fit(
                    X_train, y_train.astype("float32"), return_val=True
                )

                objective = mean_absolute_error(model.predict(X_val), y_val)
                return objective

            # 3. Create a study object and optimize the objective function.
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)
            best_params = study.best_params

            config = Config(
                verbose=False,
                cuda_if_avail=True,
                model=Config.Model(in_dim=n_features, n_hiddens=3, dropout=0),
                training=Config.Training(
                    lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"],
                    batch_size=512,
                    num_updates=5000,
                    patience=10,
                ),
            )
            model = TorchRegressor(config=config, model=MLP(args=config.model))
            model.fit(X_train, y_train.astype("float32"))
            self.model = model

        def predict(self, X_train):
            return np.nan_to_num(self.model.predict(X_train))

    return Model()


def TorchMLPClassifier():
    # 1. Define an objective function to be maximized.
    class Model:
        def __init__(self):
            pass

        def fit(self, X_train, y_train):
            print(
                f"MLP classifier got: X_train {X_train.shape}, y_train: {y_train.shape}"
            )
            n_features = X_train.shape[-1]
            n_out = y_train.max() + 1

            def objective(trial: optuna.Trial):

                # 2. Suggest values of the hyperparameters using a trial object.
                lr = trial.suggest_float("lr", high=0.1, low=0.0001, log=True)
                weight_decay = trial.suggest_float(
                    "weight_decay", high=0.1, low=0.00001, log=True
                )

                config = Config(
                    verbose=False,
                    cuda_if_avail=True,
                    model=Config.Model(
                        in_dim=n_features, out_dim=n_out, n_hiddens=3, dropout=0
                    ),
                    training=Config.Training(
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=512,
                        num_updates=5000,
                        patience=10,
                    ),
                )
                model = TorchClassifier(config=config, model=MLP(args=config.model))
                train_losses, val_losses, X_val, y_val = model.fit(
                    X_train, y_train, return_val=True
                )

                objective = accuracy_score(model.predict(X_val), y_val)
                return objective

            # 3. Create a study object and optimize the objective function.
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)
            best_params = study.best_params

            config = Config(
                verbose=False,
                cuda_if_avail=True,
                model=Config.Model(
                    in_dim=n_features, out_dim=n_out, n_hiddens=3, dropout=0
                ),
                training=Config.Training(
                    lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"],
                    batch_size=512,
                    num_updates=5000,
                    patience=10,
                ),
            )
            model = TorchClassifier(config=config, model=MLP(args=config.model))
            model.fit(X_train, y_train)
            self.model = model

        def predict(self, X_train):
            pred = np.nan_to_num(self.model.predict(X_train))
            print(f"pred: {pred.shape}")
            return pred
            # return pred.squeeze(-1)

    return Model()
