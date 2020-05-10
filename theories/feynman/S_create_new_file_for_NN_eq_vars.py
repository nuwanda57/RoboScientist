import numpy as np
from sympy.parsing.sympy_parser import parse_expr


def create_new_file_for_NN_eq_vars(pathdir, filename, newfile_name, index, equation):
    sin = np.sin
    asin = np.arcsin
    atan = np.arctan
    cos = np.cos
    sqrt = np.sqrt
    exp = np.exp
    log = np.log
    pi = np.pi
    E = np.exp(1)

    equation = str(equation)
    equation = equation.replace("Pi", "pi")
    equation = parse_expr(equation)

    n_variables = np.loadtxt(pathdir + filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + filename, usecols=(n_variables,))

    # calculate the discovered part
    v1 = str(next(iter(equation.free_symbols)))
    vars()[v1] = variables[:, index]
    solved_numerical_part = eval(str(equation))

    # get the new output
    f_dependent = f_dependent / solved_numerical_part
    variables = np.column_stack((variables, f_dependent))

    # save the new data to file
    np.savetxt(pathdir + newfile_name, variables)

    return (newfile_name)
