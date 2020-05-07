import numpy as np
import sys
import os
from copy import copy
from contextlib import redirect_stdout
from sklearn.metrics import mean_squared_error
import re
import torch
from theories import base
from theories.feynman.aiFeynman import aiFeynman
from theories.nested_formulas.nested_formula import NestedFormula, LearnFormula
import copy

# import torch
# import torch.nn as nn
# from auxiliary_functions import *

class TheoryNestedFormula(base.TheoryBase):
    def train(self, X_train, y_train):
        super().train(X_train, y_train)
        
        stdout = 'nested_formula_stdout.txt'
        if os.path.exists(stdout):
            os.remove(stdout)
        with open(stdout, 'a') as f:
            with redirect_stdout(f):
                self._logger.info('Redirecting stdout into {}'.format(stdout))
                
        self.inner_nested_formula, _ = LearnFormula(X_train, y_train, optimizer_for_formula=torch.optim.Rprop, n_init=4)
        

        formula = str(self.inner_nested_formula)
        self._logger.info('Resulting formula {}'.format(formula))
        self._formula_string = formula
        
    def calculate_test_mse(self, X_test, y_test):
        y_pred = self.inner_nested_formula.forward(X_test).detach()
        return mean_squared_error(y_test, y_pred)
