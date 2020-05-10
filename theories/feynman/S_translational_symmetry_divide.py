# checks if f(x,y)=f(x+a,y+a)
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()


class SimpleNet(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.linear1 = nn.Linear(ni, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 64)
        self.linear6 = nn.Linear(64, 64)
        self.linear7 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.softplus(self.linear1(x))
        x = F.softplus(self.linear2(x))
        x = F.softplus(self.linear3(x))
        x = F.softplus(self.linear4(x))
        x = F.softplus(self.linear5(x))
        x = F.softplus(self.linear6(x))
        x = self.linear7(x)
        return x


def rmse_loss(pred, targ):
    denom = targ ** 2
    denom = torch.sqrt(denom.sum() / len(denom))
    return torch.sqrt(F.mse_loss(pred, targ)) / denom


def translational_symmetry_divide(pathdir, filename, err_sym_divide_factor=-1):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir + "/%s" % filename, dtype='str').shape[1] - 1
        variables = np.loadtxt(pathdir + "/%s" % filename, usecols=(0,))

        if n_variables == 1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (0, 0, 0)
        else:
            for j in range(1, n_variables):
                v = np.loadtxt(pathdir + "/%s" % filename, usecols=(j,))
                variables = np.column_stack((variables, v))

        f_dependent = np.loadtxt(pathdir + "/%s" % filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent, (len(f_dependent), 1))

        factors = torch.from_numpy(variables)
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights + filename + ".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():
            if rmse_loss(model(factors), product) > 0.01:
                return (0, 0, 0)

            a = 1.2
            # make the shift x->x*a and y->y*a for 2 variables at a time (different variables)
            for i in range(0, n_variables, 1):
                for j in range(0, n_variables, 1):
                    if i < j:
                        fact_translate = factors.clone()
                        fact_translate[:, i] = fact_translate[:, i] * a
                        fact_translate[:, j] = fact_translate[:, j] * a

                        if err_sym_divide_factor == -1:
                            error_threshold = 7 * rmse_loss(model(factors), product)
                        else:
                            error_threshold = err_sym_divide_factor * rmse_loss(model(factors), product)
                        print(filename, error_threshold)
                        error = torch.sqrt(torch.mean((product - model(fact_translate)) ** 2)) / torch.sqrt(
                            torch.mean(product ** 2))

                        print("ERROR: ", abs(error))
                        if abs(error) < error_threshold:
                            file_name = filename + "-translated_divide"
                            data_translated = variables
                            data_translated[:, i] = variables[:, i] / variables[:, j]
                            data_translated = np.delete(data_translated, j, axis=1)
                            data_translated = np.column_stack((data_translated, f_dependent))
                            try:
                                os.mkdir("results/translated_data_divide/")
                            except:
                                pass
                            np.savetxt("results/translated_data_divide/" + file_name, data_translated)
                            print("SUCCESS", i, j)
                            return (file_name, i, j)

    except Exception as e:
        print(e)
        return (0, 0, 0)

    return (0, 0, 0)
