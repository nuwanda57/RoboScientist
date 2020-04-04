import numpy as np
from sympy import symbols, Add, Mul, S
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp

def create_new_file_for_NN_mult(pathdir,filename,newfile_name,indices,equation):
    sin = np.sin
    asin = np.arcsin
    atan = np.arctan
    cos = np.cos
    sqrt = np.sqrt
    exp = np.exp
    log = np.log
    pi = np.pi
    E = np.exp(1)

    equation = parse_expr(equation)

    n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
    variables = np.loadtxt(pathdir+filename, usecols=(0,))
    for j in range(1,n_variables):
        v = np.loadtxt(pathdir+filename, usecols=(j,))
        variables = np.column_stack((variables,v))
    f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))

    solved_variables = variables[:,indices[0]]
    for w in range(1,len(indices),1):
        solved_variables =np.column_stack((solved_variables,variables[:,indices[w]]))

    del_indices = []
    for w in range(len(indices)):
        del_indices = del_indices + [indices[w]-w]

    for w in del_indices:
        variables = np.delete(variables, w,axis=1)

    # calculate the discovered part
    free_symbs = sorted(equation.free_symbols, key = lambda symbol: symbol.name)
    t = 0

    for v in free_symbs:
        v = str(v)
        if len(free_symbs)>1:
            vars()[v] = solved_variables[:,t]
        else:
            vars()[v] = solved_variables
        t = t+1

    solved_numerical_part = eval(str(equation))
    # get the new output
    f_dependent = f_dependent/solved_numerical_part

    variables = np.column_stack((variables,f_dependent))

    np.savetxt(pathdir+newfile_name,variables)

    return(newfile_name)


