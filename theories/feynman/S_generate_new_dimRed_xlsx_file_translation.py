# use this to get the right variables after applying the translational a operation
import numpy as np
import pandas as pd


def generate_new_dimRed_xlsx_file_translation(original_xlsx, filename, symbol, index1, index2):
    dimRed_file = pd.read_excel(original_xlsx)
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
            columns = ["Filename", "Formula", "var1", "var2", "var3", "var4", "var5", "var6"]

            if pd.isnull(var1[i]) == 0 and var1[i] != " ":
                if var1[i][0] == ' ':
                    vars = vars + [var1[i][1:]]
                else:
                    vars = vars + [var1[i]]
            if pd.isnull(var2[i]) == 0 and var2[i] != " ":
                if var2[i][0] == ' ':
                    vars = vars + [var2[i][1:]]
                else:
                    vars = vars + [var2[i]]
            if pd.isnull(var3[i]) == 0 and var3[i] != " ":
                if var3[i][0] == ' ':
                    vars = vars + [var3[i][1:]]
                else:
                    vars = vars + [var3[i]]
            if pd.isnull(var4[i]) == 0 and var4[i] != " ":
                if var4[i][0] == ' ':
                    vars = vars + [var4[i][1:]]
                else:
                    vars = vars + [var4[i]]
            if pd.isnull(var5[i]) == 0 and var5[i] != " ":
                if var5[i][0] == ' ':
                    vars = vars + [var5[i][1:]]
                else:
                    vars = vars + [var5[i]]
            if pd.isnull(var6[i]) == 0 and var6[i] != " ":
                if var6[i][0] == ' ':
                    vars = vars + [var6[i][1:]]
                else:
                    vars = vars + [var6[i]]

            vars = np.array(vars, dtype=object)

            if symbol == "*":
                vars[index1] = vars[index1] + "*" + vars[index2]
            if symbol == "/":
                vars[index1] = vars[index1] + "/" + vars[index2]
            if symbol == "-":
                vars[index1] = vars[index1] + "-" + vars[index2]
            if symbol == "+":
                vars[index1] = vars[index1] + "+" + vars[index2]
            vars = np.delete(vars, index2)

            data = [dimRed_filename[i], dimRed_formula[i]]
            for j in range(6):
                if j < len(vars):
                    data = data + [vars[j]]
                else:
                    data = data + [""]

            new_xlsx = "results/" + filename + "-" + "translation" + ".xlsx"
            data = [data]

            df = pd.DataFrame(data, columns=columns)
            df.to_excel(new_xlsx, index=False)
            break

    return new_xlsx
