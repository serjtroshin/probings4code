from typing import List, Union

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class LinearModel(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.linear(x)


class IPDistance(nn.Module):
    def __init__(self, transform: nn.Module):
        super().__init__()
        self.transform = transform  # by default LinearModel

    def forward(self, h_i, h_j):
        # h_i, h_j \in R^{bs, emb_dim}
        vec = self.transform(h_i - h_j)
        return (vec * vec).sum(-1)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred_dists, true_dists):
        """MSE loss for pairs of vectors
        Args:
            pred_dists: distances between h_i, w_j
            dists (bs): true distances between w_i, w_j
        """
        return self.mse(pred_dists, true_dists)


class SimpleModel:
    def __init__(self, model, emb_dim):
        self.model = model

    def fit(self, h_i, h_j, true_dists):
        h_i = np.array(h_i)
        h_j = np.array(h_j)
        h_concat = np.concatenate((h_i, h_j), axis=-1)
        self.model.fit(h_concat, true_dists)

    def predict(self, h_i, h_j) -> List[Union[int, float]]:
        h_i = np.array(h_i)
        h_j = np.array(h_j)
        h_concat = np.concatenate((h_i, h_j), axis=-1)
        return self.model.predict(h_concat)


class Model:
    def __init__(self, emb_dim):
        self.bs = 64
        self.n_epochs = 100
        self.dist = IPDistance(LinearModel(emb_dim))
        self.metric = MSELoss()
        self.optimizer = Adam(self.dist.parameters())

    def fit(self, h_i, h_j, true_dists):
        h_i = torch.tensor(h_i)
        h_j = torch.tensor(h_j)
        true_dists = torch.FloatTensor(true_dists)

        for n_epoch in range(self.n_epochs):
            with tqdm(range(0, h_i.shape[0] - self.bs, self.bs)) as pbar:
                for i in pbar:
                    pred_dists = self.dist(h_i[i : i + self.bs], h_j[i : i + self.bs])
                    loss = self.metric(pred_dists, true_dists[i + self.bs])

                    pbar.set_description(
                        f"n_epoch> {n_epoch} loss train {loss.item():.4f}"
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def predict(self, h_i, h_j) -> List[Union[int, float]]:
        h_i = torch.tensor(h_i)
        h_j = torch.tensor(h_j)

        preds = []
        with torch.no_grad():
            for i in range(h_i.shape[0]):
                preds.append(self.dist(h_i[i : i + 1], h_j[i : i + 1])[0].item())
        return preds
