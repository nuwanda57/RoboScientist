from __future__ import print_function

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


# pass the indices to be made equal
def NN_equal_vars(pathdir, filename, i, j):
    pathdir_weights = "results/NN_trained_models/models/"

    # load the data
    n_variables = np.loadtxt(pathdir + "/%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "/%s" % filename, usecols=(0,))

    if n_variables == 1:
        print(filename, "just one variable for ADD")
        # if there is just one variable you have nothing to separate
        return (0, 0)
    else:
        for j in range(1, n_variables):
            v = np.loadtxt(pathdir + "/%s" % filename, usecols=(j,))
            variables = np.column_stack((variables, v))

    f_dependent = np.loadtxt(pathdir + "/%s" % filename, usecols=(n_variables,))
    f_dependent = np.reshape(f_dependent, (len(f_dependent), 1))

    factors = torch.from_numpy(variables[0:10000])
    if is_cuda:
        factors = factors.cuda()
    else:
        factors = factors
    factors = factors.float()

    product = torch.from_numpy(f_dependent[0:10000])
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

    with torch.no_grad():
        fact_equal = factors.clone()
        fact_equal[:, j] = fact_equal[:, i]
        model_equal = model(fact_equal)
        data_equal = np.delete(fact_equal.cpu(), i, axis=1)
        data_equal = np.column_stack((data_equal.cpu(), model_equal.cpu()))
        file_name = filename + "-eq_var_%s_%s" % (i, j)
        np.savetxt("results/equal_variables/" + file_name, data_equal)

        return (file_name)
