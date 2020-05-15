import pickle

import torch
import copy

from theories import base, theory_nested_formulas


class TheoryMultipleNestedFormulas(base.TheoryBase):
    def __init__(self, *args):
        super(TheoryMultipleNestedFormulas, self).__init__(*args)
        self.n_formulas = 10
        self._all_models = []
        self._model = None

    def train(self, X, y, optimizer_for_formula=torch.optim.Adam, device=torch.device("cpu"), n_init=1,
              max_iter=100,
              lr=0.01,
              depth=1, verbose_frequency=5000,
              max_epochs_without_improvement=1000,
              minimal_acceptable_improvement=1e-6, max_tol=1e-5):

        """
        Parameters:
            X: torch.tensor, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y: torch.tensor, shape (n_samples, 1)
                Target vector relative to X.
            optimizer_for_formula: 
                optimizer used for solving the optimization problem
            device:
                device used to store the data and run the algorithm
            n_init: int
                number of times algorithm will be run with different initial weights.
                The final results will be the best output of n_init consecutive runs in terms of loss.
            max_iter: int
                Maximum number of iterations of the algorithm for a single run.
            depth: int
                depth of formula to learn
            verbose_frequency: int
                print loss every verbose_frequency epochs
            max_epochs_without_improvement: int
                if during this number of epochs loss does not decrease more than minimal_acceptable_improvement, the learning process
                will be finished
            minimal_acceptable_improvement: float
                if during max_epochs_without_improvement number of epochs loss does not decrease more than this number,
                the learning process will be finished
            max_tol: float
                if the loss becomes smaller than this value, stop performing initializations and finish the learning process
        """
        super().train(X, y)
        smallest_mse = 1e50
        for model_number in range(self.n_formulas):
            single_theory = theory_nested_formulas.TheoryNestedFormula()
            single_theory.train(X, y)
            current_mse = single_theory.calculate_test_mse(X, y)
            if current_mse < smallest_mse:
                current_mse = smallest_mse
                self._model = single_theory
            self._all_models.append(single_theory)

        self._formula_string = str(self._model)
        self._logger.info('Resulting formula {}'.format(self._formula_string))

    def calculate_test_mse(self, X_test, y_test):
        if self._model is None:
            self._logger.info('Theory is not trained.')
            return 1000
        return self._model.calculate_test_mse(X_test, y_test)

    def __deepcopy__(self, memodict={}):
        new_obj = super().__deepcopy__(memodict)
        return new_obj

    def std(self, x):
        predictions = []
        for model_number in range(self.n_formulas):
            predictions.append(self._all_models[model_number]._model.forward(x).detach())
        return torch.stack(predictions).std(axis=0)
