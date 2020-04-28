import numpy as np
import sys
import os
from copy import copy
from contextlib import redirect_stdout
from sklearn.metrics import mean_squared_error
import re

from theories import base
from theories.feynman.aiFeynman import aiFeynman


class TheoryFeynman(base.TheoryBase):
    def train(self, X_train, y_train):
        file_data = np.array([X_train.numpy(), y_train.numpy()]).T

        filename = '001.a'
        np.savetxt('./data/' + filename, file_data)

        feinman_stdout = 'feinman_stdout.txt'
        if os.path.exists(feinman_stdout):
            os.remove(feinman_stdout)
        with open(feinman_stdout, 'a') as f:
            with redirect_stdout(f):
                aiFeynman('./data/' + filename)

        solved_file = open("results/solutions/" + filename + '.txt')

        self._formula_string = solved_file.readlines()[0].split()[1]
        
    def calculate_test_mse(self, X_test, y_test):
        f = copy(self._formula_string)
        f = f.replace('sqrt', 'np.sqrt').replace('exp', 'np.exp')\
            .replace('pi', 'np.pi').replace('sin', 'np.sin').replace('log', 'np.log')

        try:
            pred = [eval(re.sub(r'[^a-zA-Z]x[^a-zA-Z]', str(x.item()), f)) for x in X_test]
            return mean_squared_error(pred, y_test)
        except:
            return 1000
#         return None
