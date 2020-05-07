import torch
from sklearn.metrics import mean_squared_error

from theories import base
from theories.nested_formulas.nested_formula import NestedFormula


import torch
import torch.nn as nn
# from auxiliary_functions import *


class TheoryNestedFormula(base.TheoryBase):

    def train(self, X, y, optimizer_for_formula=torch.optim.Adam, device=torch.device("cpu"), n_init=1,
              max_iter=100,
              lr=0.01,
              depth=1, verbose=2, verbose_frequency=5000,
              max_epochs_without_improvement=1000,
              minimal_acceptable_improvement=1e-6, max_tol=1e-5):

        """
        Parameters:
            X: torch.tensor, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y: torch.tensor, shape (n_samples, 1)
                Target vector relative to X.
            n_init: int
                number of times algorithm will be run with different initial weights.
                The final results will be the best output of n_init consecutive runs in terms of loss.
            max_iter: int
                Maximum number of iterations of the algorithm for a single run.
            depth: int
                depth of formula to learn
            verbose: int
                if is equal to 0, no output
                if is equal to 1, output number of runs and losses
                if is equal to 2, output number of runs and losses and print loss every verbose_frequency epochs
            verbose_frequency: int
                if verbose equals 2, then print loss every verbose_frequency epochs
            max_epochs_without_improvement: int
                if during this number of epochs loss does not decrease more than minimal_acceptable_improvement, the learning process
                will be finished
            minimal_acceptable_improvement: float
                if during max_epochs_without_improvement number of epochs loss does not decrease more than this number,
                the learning process will be finished
            max_tol: float
                if the loss becomes smaller than this value, stop performing initializations and finish the learning process

        Returns:
            best_formula: RecursiveFormula
                fitted formula
            best_losses: list of float
                loss values for best initialization
        """
        super().train(X, y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        m = X.shape[1]
        best_formula = NestedFormula(depth, m).to(device)
        best_loss = 1e20
        best_losses = []

        for init in range(n_init):
            losses = []
            if verbose > 0:
                print("  Initialization #{}".format(init + 1))
            #     torch.random.manual_seed(seed)
            model = NestedFormula(depth, m).to(device)

            criterion = nn.MSELoss()
            epochs_without_improvement = 0
            epoch = 0
            output = model(X)
            previous_loss = criterion(output, y).item()

            optimizer = optimizer_for_formula(model.parameters(), lr)
            optimizer.zero_grad()

            while epoch < max_iter and epochs_without_improvement < max_epochs_without_improvement:
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                losses.append(loss.item())
                loss.backward()
                if verbose == 2 and (epoch + 1) % verbose_frequency == 0:
                    print("    Epoch {}, current loss {:.3}, current formula ".format(epoch + 1, loss.item()), end='')
                    PrintFormula(model, "fast")
                optimizer.step()
                epoch += 1
                if torch.abs(previous_loss - loss) < minimal_acceptable_improvement:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0
                previous_loss = loss.item()
                if epoch == 1000 and loss > 1e5:
                    print("  The model does not seem to converge, finishing at epoch 1000")
                    epoch = max_iter
            if loss < best_loss:
                best_loss = loss
                best_formula = model
                best_losses = losses
            if verbose > 0:
                print("  Finished run #{}, loss {}, best loss {}".format(init + 1, loss, best_loss))
            if loss < max_tol:
                print(f'loss is smaller than {max_tol}, terminating learning process')
                break

        formula = str(best_formula)
        self.inner_nested_formula = best_formula
        self._logger.info('Resulting formula {}'.format(formula))
        self._formula_string = formula


def calculate_test_mse(self, X_test, y_test):
    y_pred = self.inner_nested_formula.forward(X_test).detach()
    return mean_squared_error(y_test, y_pred)
