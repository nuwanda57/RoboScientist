import numpy as np


def get_RMS(X_train, y_train):
    n_variables = X_train.shape[1]
    # n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
    # f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))

    rms = np.sqrt(np.mean(X_train**2))

    # np.savetxt("results/RMS_file_value.txt",[rms], fmt='%f')
    #
    # return 1

    return rms


