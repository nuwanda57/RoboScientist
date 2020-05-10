import os

import numpy as np


def input_divide_2(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = 1 / 2 * variables
    else:
        variables[:, var_index] = 1 / 2 * variables[:, var_index]
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_divide_2/")
    except:
        pass

    fn = filename + "-input_divide_2"
    np.savetxt("results/mystery_world_input_divide_2/" + fn, variables)

    return fn


def input_multiply_2(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = 2 * variables
    else:
        variables[:, var_index] = 2 * variables[:, var_index]
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_multiply_2/")
    except:
        pass

    fn = filename + "-input_multiply_2"
    np.savetxt("results/mystery_world_input_multiply_2/" + fn, variables)

    return fn


def input_exp(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.exp(variables)
    else:
        variables[:, var_index] = np.exp(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_exp/")
    except:
        pass

    fn = filename + "-input_exp"
    np.savetxt("results/mystery_world_input_exp/" + fn, variables)

    return fn


def input_log(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.log(variables)
    else:
        variables[:, var_index] = np.log(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_log/")
    except:
        pass

    fn = filename + "-input_log"
    np.savetxt("results/mystery_world_input_log/" + fn, variables)

    return fn


def input_sqrt(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.sqrt(variables)
    else:
        variables[:, var_index] = np.sqrt(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_sqrt/")
    except:
        pass

    fn = filename + "-input_sqrt"
    np.savetxt("results/mystery_world_input_sqrt/" + fn, variables)

    return fn


def input_squared(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = variables ** 2
    else:
        variables[:, var_index] = variables[:, var_index] ** 2
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_squared/")
    except:
        pass

    fn = filename + "-input_squared"
    np.savetxt("results/mystery_world_input_squared/" + fn, variables)

    return fn


def input_inverse(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = 1 / variables
    else:
        variables[:, var_index] = 1 / variables[:, var_index]
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_inverse/")
    except:
        pass

    fn = filename + "-input_inverse"
    np.savetxt("results/mystery_world_input_inverse/" + fn, variables)

    return fn


def input_sin(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.sin(variables)
    else:
        variables[:, var_index] = np.sin(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_sin/")
    except:
        pass

    fn = filename + "-input_sin"
    np.savetxt("results/mystery_world_input_sin/" + fn, variables)

    return fn


def input_asin(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.arcsin(variables)
    else:
        variables[:, var_index] = np.arcsin(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_asin/")
    except:
        pass

    fn = filename + "-input_asin"
    np.savetxt("results/mystery_world_input_asin/" + fn, variables)

    return fn


def input_cos(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    try:
        os.mkdir("results/mystery_world_input_cos/")
    except:
        pass

    if n_variables == 1:
        variables = np.cos(variables)
    else:
        variables[:, var_index] = np.cos(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    fn = filename + "-input_cos"
    np.savetxt("results/mystery_world_input_cos/" + fn, variables)

    return fn


def input_acos(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.arccos(variables)
    else:
        variables[:, var_index] = np.arccos(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_acos/")
    except:
        pass

    fn = filename + "-input_acos"
    np.savetxt("results/mystery_world_input_acos/" + fn, variables)

    return fn


def input_tan(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.tan(variables)
    else:
        variables[:, var_index] = np.tan(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_tan/")
    except:
        pass

    fn = filename + "-input_tan"
    np.savetxt("results/mystery_world_input_tan/" + fn, variables)

    return fn


def input_atan(pathdir, filename, var_index):
    n_variables = np.loadtxt(pathdir + "%s" % filename, dtype='str').shape[1] - 1
    variables = np.loadtxt(pathdir + "%s" % filename, usecols=(0,))
    for j in range(1, n_variables):
        v = np.loadtxt(pathdir + "%s" % filename, usecols=(j,))
        variables = np.column_stack((variables, v))
    f_dependent = np.loadtxt(pathdir + "%s" % filename, usecols=(n_variables,))

    if n_variables == 1:
        variables = np.arctan(variables)
    else:
        variables[:, var_index] = np.arctan(variables[:, var_index])
    variables = np.column_stack((variables, f_dependent))

    try:
        os.mkdir("results/mystery_world_input_atan/")
    except:
        pass

    fn = filename + "-input_atan"
    np.savetxt("results/mystery_world_input_atan/" + fn, variables)

    return fn
