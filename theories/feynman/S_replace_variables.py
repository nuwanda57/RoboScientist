# this function gets the reduced variables for a given equation e.g. u/v, m2/m1 etc.

import pandas as pd
from sympy.parsing.sympy_parser import parse_expr


def replace_variables(input_file, filename, formula):
    # clean weird things in the formula such as Pi -> pi not to be seen as a variable
    formula = str(formula)
    formula = formula.replace("Pi", "pi")
    formula = parse_expr(formula)

    dimRed_file = pd.read_excel(input_file)
    dimRed_formula = dimRed_file["Formula"]
    dimRed_filename = dimRed_file["Filename"]
    var1 = dimRed_file["var1"]
    var2 = dimRed_file["var2"]
    var3 = dimRed_file["var3"]
    var4 = dimRed_file["var4"]
    var5 = dimRed_file["var5"]
    var6 = dimRed_file["var6"]

    # get the real dimensional reduced variables
    vars = []
    for i in range(len(dimRed_filename)):
        if dimRed_filename[i] == filename:
            if pd.isnull(var1[i]) == 0:
                if var1[i][0] != " ":
                    if var1[i] != '':
                        vars = vars + [var1[i]]
                else:
                    if var1[i][1:] != '':
                        vars = vars + [var1[i][1:]]
            if pd.isnull(var2[i]) == 0:
                if var2[i][0] != " ":
                    if var2[i] != '':
                        vars = vars + [var2[i]]
                else:
                    if var2[i][1:] != '':
                        vars = vars + [var2[i][1:]]
            if pd.isnull(var3[i]) == 0:
                if var3[i][0] != " ":
                    if var3[i] != '':
                        vars = vars + [var3[i]]
                else:
                    if var3[i][1:] != '':
                        vars = vars + [var3[i][1:]]
            if pd.isnull(var4[i]) == 0:
                if var4[i][0] != " ":
                    if var4[i] != '':
                        vars = vars + [var4[i]]
                else:
                    if var4[i][1:] != '':
                        vars = vars + [var4[i][1:]]
            if pd.isnull(var5[i]) == 0:
                if var5[i][0] != " ":
                    if var5[i] != '':
                        vars = vars + [var5[i]]
                else:
                    if var5[i][1:] != '':
                        vars = vars + [var5[i][1:]]
            if pd.isnull(var6[i]) == 0:
                if var6[i][0] != " ":
                    if var6[i] != '':
                        vars = vars + [var6[i]]
                else:
                    if var6[i][1:] != '':
                        vars = vars + [var6[i][1:]]
            break

    # get the discovered symbols and repalce them with the original ones
    discovered_symbols = sorted(formula.free_symbols, key=lambda symbol: symbol.name)
    for i in range(len(vars)):
        formula = formula.subs(discovered_symbols[i], "(" + vars[i] + ")")

    return formula
