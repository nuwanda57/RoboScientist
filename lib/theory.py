import lib.models as models
import torch.nn as nn
import torch
import torch.optim as optim


class Theory(object):
    def __init__(self, params_cnt=1, model=models.SingleParameterFormula(1)):
        self._params_cnt = params_cnt
        self._model = model
        self._formula_string = None

    def train(self, X_train, y_train):
        optimizer = optim.Adam(self._model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        loss = None
        for epoch in range(100):
            optimizer.zero_grad()
            net_out = self._model(X_train)
            loss = criterion(net_out, y_train)
            loss.backward()
            optimizer.step()
        return loss

    def calculate_test_mse(self, X_test, y_test):
        self._model.eval()
        with torch.no_grad():
            criterion = nn.MSELoss()
            return criterion(self._model(X_test), y_test)

    def show_model(self):
        print(self._model)
