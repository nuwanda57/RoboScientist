import numpy as np
import os

def get_squared(pathdir,pathdir_write_to,filename):
    try:
        os.mkdir(pathdir_write_to)
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
            variables = np.column_stack((variables,v))
        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))

        if n_variables==1:
            f = open(pathdir_write_to+filename+"-squared","w")
            for i in range(len(variables)):
                f.write(str(variables[i]))
                f.write(" ")
                f.write(str(f_dependent[i]**2))
                f.write("\n")
                    
        if n_variables>1:
            f = open(pathdir_write_to+filename+"-squared","w")
            for i in range(len(variables)):
                for j in range(len(variables[j])):
                    f.write(str(variables[i][j]))
                    f.write(" ")
                f.write(str(f_dependent[i]**2))
                f.write("\n")

    except:
        return 0

    return 1

