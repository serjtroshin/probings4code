from dataclasses import dataclass
from itertools import cycle
from typing import Union

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm


def infinite_loop(data_loader):
    generator = iter(data_loader)
    while True:
        try:
            # Samples the batch
            elem = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator = iter(data_loader)
            elem = next(generator)
        yield elem


# import matplotlib.pyplot as plt
# import matplotlib
# import logging
# import matplotlib
# matplotlib._log.disabled = True
# # logging.basicConfig(level='critical')
# # Turn off sina logging
# for name in [
#              "matplotlib", "matplotlib.font", "matplotlib.pyplot"]:
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.CRITICAL)
#     logger.disabled = True

# def plot(train_losses, val_losses, tag="default"):
#     p = plt.plot([i for i, _ in train_losses], [m for _, m in train_losses], alpha=0.3)
#     plt.plot([i for i, _ in val_losses], [m for _, m in val_losses], label=f"val_{tag}", marker="*", color=p[0].get_color())
#     plt.yscale("log")
#     plt.ylabel("loss")
#     plt.legend()


from collections import deque


# thanks to https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class NumpyDataset(TensorDataset):
    def __init__(self, X, y=None, cuda_if_avail=False):
        def downcast_if(x: torch.Tensor) -> torch.Tensor:
            if isinstance(x, torch.DoubleTensor):
                x = x.float()
            #             if isinstance(x, torch.LongTensor):
            #                 x = x.int()
            return x

        def unsqeeze_if(x):
            if len(x.shape) == 1:
                return torch.unsqueeze(x, -1)
            return x

        X = downcast_if(torch.from_numpy(X))
        if y is not None:
            y = unsqeeze_if(downcast_if(torch.from_numpy(y)))
        if torch.cuda.is_available() and cuda_if_avail:
            X = X.cuda()
            if y is not None:
                y = y.cuda()
        if y is None:
            super(NumpyDataset, self).__init__(
                X,
            )
        else:
            super(NumpyDataset, self).__init__(X, y)


@dataclass
class Config:
    @dataclass
    class Training:
        lr: Union[float, int]
        weight_decay: float
        batch_size: int

        num_updates: int = 100
        valid_every: int = 100
        num_updates: int = 1000
        optimizer: str = "AdamW"
        early_stoping: bool = True
        patience: int = 3

    @dataclass
    class Model:
        in_dim: int = 10
        hidden_dim: int = 128
        out_dim: int = 1
        n_hiddens: int = 1
        dropout: float = 0.0

    training: Training
    model: Model = Model()

    seed: int = 1
    verbose: bool = True
    cuda_if_avail: bool = True
    eval_batch_size: int = 8192


class MLP(nn.Module):
    def __init__(self, args: Config.Model):
        super(MLP, self).__init__()

        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.n_hiddens = args.n_hiddens
        self.dropout = args.dropout

        if self.n_hiddens == 0:
            self.nn = nn.Linear(self.in_dim, self.out_dim)
        else:
            layers = [
                nn.Dropout(self.dropout),
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
            ]
            for i in range(self.n_hiddens - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, self.out_dim))
            self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class Model(nn.Module):
    def __init__(self, model: nn.Module, loss: nn.Module, config: Config):
        super(Model, self).__init__()
        self.model = model
        self.loss = loss
        self.args = config.training
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=int(self.args.patience / 2)
        )
        self.early_stopping = EarlyStopping(mode="min", patience=self.args.patience)

    def validation(self, valid_loader: DataLoader) -> float:
        self.eval()
        loss = 0.0
        n = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                inp, labels = batch
                pred = self.model(inp)
                loss = loss + self.loss(pred.squeeze(-1), labels.squeeze(-1))
                n += 1
        return loss / n

    def train_num_updates(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_updates: int,
        valid_every: int,
    ):
        train_losses = []
        valid_losses = []
        for i, batch in enumerate(cycle(train_loader)):
            self.train()
            if i == num_updates:
                break
            #             print(batch)
            inp, labels = batch
            pred = self.model(inp)

            loss = self.loss(pred.squeeze(-1), labels.squeeze(-1)).mean(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % valid_every == valid_every - 1:
                valid_loss = self.validation(valid_loader)
                self.lr_sheduler.step(valid_loss)
                valid_losses.append((i, valid_loss.item()))
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if self.config.verbose:
                    print(
                        f"num_update: {i}: valid_loss: {valid_loss: 0.5f}, lr: {curr_lr: 0.7f}"
                    )

                if self.early_stopping.step(valid_loss):
                    if self.config.verbose:
                        print("early stopping")
                    break

            train_losses.append((i, loss.item()))
        return train_losses, valid_losses

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, valid_size=0.1, return_val=False
    ):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_size
        )
        if self.config.verbose:
            print(
                "dataset:", X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
            )
        train_loader = infinite_loop(
            DataLoader(
                NumpyDataset(X_train, y_train, cuda_if_avail=self.config.cuda_if_avail),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
        )
        valid_loader = DataLoader(
            NumpyDataset(X_valid, y_valid, cuda_if_avail=self.config.cuda_if_avail),
            batch_size=self.config.eval_batch_size,
        )
        train_losses, valid_losses = self.train_num_updates(
            train_loader, valid_loader, self.args.num_updates, self.args.valid_every
        )
        if return_val:
            return train_losses, valid_losses, X_valid, y_valid
        return train_losses, valid_losses


class TorchRegressor(Model):
    def __init__(self, model: nn.Module, config: Config):
        """
        fits and validates a backbone + MLELoss,
        Adam(lr) + ReduceLROnPlateau + EarlyStopping(patience)
        """
        super(TorchRegressor, self).__init__(
            model=model, loss=nn.MSELoss(), config=config
        )
        if self.config.verbose:
            print(self.config)
            print(self)

        if torch.cuda.is_available() and self.config.cuda_if_avail:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            loader = DataLoader(
                NumpyDataset(X, cuda_if_avail=self.config.cuda_if_avail),
                batch_size=8,
                drop_last=False,
            )
            preds = []
            for (batch,) in loader:
                pred = self.model(batch)
                preds.append(pred)
            return torch.cat(preds, axis=0).detach().cpu().numpy()


class TorchClassifier(Model):
    def __init__(self, model: nn.Module, config: Config):
        """
        fits and validates a backbone + CrossEntropyLoss,
        Adam(lr) + ReduceLROnPlateau + EarlyStopping(patience)
        """
        super(TorchClassifier, self).__init__(
            model=model, loss=nn.CrossEntropyLoss(), config=config
        )
        if self.config.verbose:
            print(self.config)
            print(self)

        if torch.cuda.is_available() and self.config.cuda_if_avail:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            loader = DataLoader(
                NumpyDataset(X, cuda_if_avail=self.config.cuda_if_avail),
                batch_size=8,
                drop_last=False,
            )
            preds = []
            for (batch,) in loader:
                pred = self.model(batch)
                pred = torch.argmax(pred, dim=-1)
                preds.append(pred)
            return torch.cat(preds, axis=0).detach().cpu().numpy()
