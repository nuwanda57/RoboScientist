import torch
import copy

from theories import base, theory_nested_formulas


class MasterTheory(base.TheoryBase):
    """
    Class used for representing theories that themselves consist from multiple theories

    Attributes:
        n_models
        _models
        _model - model from _models that has shown the smallest mse on train dataset
    """
    def __init__(self, *args, theories_with_params=[(theory_nested_formulas.TheoryNestedFormula, None) for i in range(10)]):
        super(MasterTheory, self).__init__(*args)
        self._models = [theory(*params) if params is not None else theory() for theory, params in theories_with_params]
        self.n_models = len(self._models)
        self._best_model = None

    def train(self, X, y):
        super().train(X, y)
        smallest_mse = 1e50
        for model_number, model in enumerate(self._models):
            model.train(X, y)
            current_mse = model.calculate_test_mse(X, y)
            if current_mse < smallest_mse:
                current_mse = smallest_mse
                self._best_model = copy.deepcopy(model)

        self._formula_string = str(self._best_model)
        self._logger.info('Resulting formula {}'.format(self._formula_string))

    def calculate_test_mse(self, X_test, y_test):
        # When we make predictions we use one theory that has shown the smallest mse on the train dataset
        if self._best_model is None:
            self._logger.info('Theory is not trained.')
            return 1000
        return self._best_model.calculate_test_mse(X_test, y_test)

    def __deepcopy__(self, memodict={}):
        new_obj = super().__deepcopy__(memodict)
        new_obj._models = copy.deepcopy(self._models)
        new_obj._model = copy.deepcopy(self._best_model)
        return new_obj

    def std(self, x):
        # We make predictions on the given dataset using all models that we have, and then take the variance
        predictions = []
        for model_number in range(self.n_models):
            predictions.append(self._models[model_number]._model.forward(x).detach())
        return torch.stack(predictions).std(axis=0)
