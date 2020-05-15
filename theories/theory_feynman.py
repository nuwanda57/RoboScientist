import numpy as np
import os
from copy import copy
from contextlib import redirect_stdout
from sklearn.metrics import mean_squared_error
import re

from theories import base
from theories.feynman.aiFeynman import aiFeynman


class TheoryFeynman(base.TheoryBase):
    def train(self, X_train, y_train):
        super().train(X_train, y_train)
        file_data = np.array([X_train.numpy(), y_train.numpy()]).T

        filename = '001.a'
        np.savetxt('./data/' + filename, file_data)

        feinman_stdout = 'feinman_stdout.txt'
        if os.path.exists(feinman_stdout):
            os.remove(feinman_stdout)
        with open(feinman_stdout, 'a') as f:
            with redirect_stdout(f):
                self._logger.info('Redirecting stdout into {}'.format(feinman_stdout))
                aiFeynman('./data/' + filename)

        solved_file = open("results/solutions/" + filename + '.txt')

        text = solved_file.readlines()[0].split()
        self._logger.info('Solved file content: {}'.format(text))
        text.pop(0)
        right = 0
        for i in range(len(text)):
            t = text[i]
            if t[0] == '[':
                right = i
                break
        formula = ''.join(text[:right])
        self._logger.info('Resulting formula {}'.format(formula))
        self._formula_string = formula
        
    def calculate_test_mse(self, X_test, y_test):
        f = copy(self._formula_string)
        f = f.replace('sqrt', 'np.sqrt').replace('exp', 'np.exp')\
            .replace('pi', 'np.pi').replace('sin', 'np.sin').replace('log', 'np.log').replace('cos', 'np.cos')

        self._logger.info('Trying to evaluate formula: {}.'.format(
            re.sub(r'([^a-zA-Z])x([^a-zA-Z])', r'\1 %f \2' % X_test[0].item(), f)))
        try:
            pred = [eval(re.sub(r'([^a-zA-Z])x([^a-zA-Z])', r'\1 %f \2' % x.item(), f)) for x in X_test]
            self._logger.info('Predicted: {}'.format(pred))
            mse = mean_squared_error(pred, y_test)
            if np.isnan(mse):
                self._logger.info('MSE is None')
                return 1000
            self._logger.info('MSE: {}'.format(mse))
            return mse
        except Exception as error:
            self._logger.error('Unable to evaluate formula {}. MSE=1000'.format(self._formula_string))
            self._logger.error('Exception raised: {}'.format(str(error)))
            return 1000
