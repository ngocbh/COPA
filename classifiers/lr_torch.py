import numpy as np
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, \
    classification_report, confusion_matrix
from dice_ml.model_interfaces.base_model import BaseModel

from torch.nn import functional as F


class LogisticRegression(torch.nn.Module):
    def __init__(self, n, data_transformer, predict_threshold):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n, 1)
        self.data_transformer = data_transformer
        self.n = n
        self.predict_threshold = predict_threshold

    def forward(self, x):
        """forward.

        Parameters
        ----------
        x : (batch_size, num_features)
            x: input
        """
        return torch.sigmoid(self.linear(x))

    def predict_proba(self, inp, transform_data=True, model_score=True):
        if transform_data:
            inp = self.data_transformer.transform(inp, tonumpy=True)
        inp = torch.tensor(inp, dtype=torch.float32)
        return self.forward(inp).detach().squeeze().numpy()

    def predict(self, inp, transform_data=True, model_score=True):
        y_pred = self.predict_proba(inp, transform_data, model_score)
        y_pred = (y_pred >= self.predict_threshold ).astype(np.float64)
        return y_pred


class LRTorch:
    backend = 'PYT'

    def __init__(self, data_transformer, lr=0.005, **kwargs):
        self.data_transformer = data_transformer
        self.lr = lr
        self.weight_decay = 1e-3
        self.num_stable_iter = 0
        self.max_stable_iter = 5
        self.loss_diff_threshold = 1e-5
        self.predict_threshold = 0.5
        self.model = None
        super(LRTorch, self).__init__(**kwargs)

    def __check_termination(self, loss_diff):
        if loss_diff <= self.loss_diff_threshold:
            self.num_stable_iter += 1
            return (self.num_stable_iter >= self.max_stable_iter)
        else:
            self.num_stable_iter = 0
            return False

    def train(self, X_train, y_train, max_iter=200, transform_data=True, verbose=False, **kwargs):
        self.train_data = (X_train, y_train)

        if transform_data:
            X_train = self.data_transformer.transform(X_train)

        _, n = X_train.shape

        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

        model = LogisticRegression(n, self.data_transformer, self.predict_threshold)
        loss_func = torch.nn.BCELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

        # print(model.linear.weight)
        loss_diff = 1.0
        prev_loss = 0.0
        self.num_stable_iter = 0

        for i in range(max_iter):
            y_pred = model(X_train)
            loss = loss_func(y_pred.squeeze(), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print("Iter %d: loss: %f" %(i, loss.data.item()))

            loss_diff = prev_loss - loss.data.item()
            if self.__check_termination(loss_diff):
                break

            prev_loss = loss.data.item()

        # print(model.linear.weight)
        self.model = model

    def set_weights(self, weights):
        w, b = weights[1:], weights[0]
        n = len(weights)
        if self.model is None:
            self.model = LogisticRegression(n, self.data_transformer, self.predict_threshold)
        w = torch.nn.Parameter(torch.tensor(w).expand_as(self.model.linear.weight.data).float())
        b = torch.nn.Parameter(torch.tensor(b).expand_as(self.model.linear.bias.data).float())
        self.model.linear.weight = w
        self.model.linear.bias = b

    @property
    def weights(self):
        """weights.

            return: [b, w] weight and bias
        """
        weights = torch.cat((self.model.linear.bias.data,
                             self.model.linear.weight.data.squeeze()))
        return weights.numpy()

    def get_model(self):
        return self.model

    def predict_proba(self, inp, transform_data=True, model_score=True):
        return self.model.predict_proba(inp, transform_data, model_score)

    def predict(self, inp, transform_data=True, model_score=True):
        return self.model.predict(inp, transform_data, model_score)

    def print_performance(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy = ", accuracy)

        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nPrecision/Recall:\n", classification_report(y_test, y_pred))
