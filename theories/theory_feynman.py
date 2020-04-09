import numpy as np

from theories import theory_base
from theories.feynman.aiFeynman import aiFeynman


class TheoryFeynman(theory_base.TheoryBase):
    def train(self, X_train, y_train):
        file_data = np.array([X_train.numpy(), y_train.numpy()]).T

        filename = '001.a'
        np.savetxt('./data/' + filename, file_data)

        aiFeynman('./data/' + filename)

        solved_file = open("results/solutions/" + filename + '.txt')

        self._formula_string = solved_file.readlines()[0].split()[1]

        return self._formula_string
