import numpy as np
import os
from baselines.ai_feynman.S_multipolyfit import getBest
from baselines.ai_feynman.S_multipolyfit import basis_vector
import itertools
import sys
import csv
from sympy import symbols, Add, Mul, S


def mk_sympy_function(coeffs, num_covariates, deg):
    generators = [basis_vector(num_covariates+1, i) for i in range(num_covariates+1)]
    powers = map(sum, itertools.combinations_with_replacement(generators, deg))
    
    coeffs = np.round(coeffs,2)
    
    xs = (S.One,) + symbols('x0:%d'%num_covariates)
    if len(coeffs)>1:
        return Add(*[coeff * Mul(*[x**deg for x, deg in zip(xs, power)])
                     for power, coeff in zip(powers, coeffs)])
    else:
        return coeffs[0]

def polyfit(maxdeg, filename, error_treshold):
    n_variables = np.loadtxt(filename, dtype='str').shape[1]-1
    variables = np.loadtxt(filename, usecols=(0,))
    for j in range(1,n_variables):
        v = np.loadtxt(filename, usecols=(j,))
        variables = np.column_stack((variables,v))
    f_dependent = np.loadtxt(filename, usecols=(n_variables,))

    print("Number of variables: ", len(variables))

    parameters = getBest(variables,f_dependent,maxdeg)[0]
    params_error = getBest(variables,f_dependent,maxdeg)[1]
    deg = getBest(variables,f_dependent,maxdeg)[2]
    print("Error: ", params_error, mk_sympy_function(parameters,n_variables,deg))
    if(params_error<error_treshold):
        return (1, mk_sympy_function(parameters,n_variables,deg))
    else:
        return (0, mk_sympy_function(parameters,n_variables,deg))

