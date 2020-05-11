import numpy as np

def get_RMS(pathdir,filename):


    n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
    f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))

    rms = np.sqrt(np.mean(f_dependent**2))

    np.savetxt("results/RMS_file_value.txt",[rms], fmt='%f')

    return 1



