# use this to get the right variables after applying the translational a operation
import numpy as np
import pandas as pd


def generate_new_dimRed_xlsx_file_separable(original_xlsx, filename, indices, side):
    fn = filename
    filename = filename.split('-')[0]

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

            if pd.isnull(var1[i]) == 0:
                if var1[i][0] == '':
                    vars = vars + [var1[i][1:]]
                else:
                    vars = vars + [var1[i]]
            if pd.isnull(var2[i]) == 0:
                if var2[i][0] == '':
                    vars = vars + [var2[i][1:]]
                else:
                    vars = vars + [var2[i]]
            if pd.isnull(var3[i]) == 0:
                if var3[i][0] == '':
                    vars = vars + [var3[i][1:]]
                else:
                    vars = vars + [var3[i]]
            if pd.isnull(var4[i]) == 0:
                if var4[i][0] == '':
                    vars = vars + [var4[i][1:]]
                else:
                    vars = vars + [var4[i]]
            if pd.isnull(var5[i]) == 0:
                if var5[i][0] == '':
                    vars = vars + [var5[i][1:]]
                else:
                    vars = vars + [var5[i]]
            if pd.isnull(var6[i]) == 0:
                if var6[i][0] == '':
                    vars = vars + [var6[i][1:]]
                else:
                    vars = vars + [var6[i]]

            vars = np.array(vars, dtype=object)

            idx = []
            for w in range(len(indices)):
                idx = idx + [indices[w] - w]

            for w in idx:
                vars = np.delete(vars, w)

            data = [dimRed_filename[i], dimRed_formula[i]]
            for j in range(6):
                if j < len(vars):
                    data = data + [vars[j]]
                else:
                    data = data + [""]

            if side == "left":
                new_xlsx = "results/" + fn + "-" + "separable_left" + ".xlsx"
            if side == "right":
                new_xlsx = "results/" + fn + "-" + "separable_right" + ".xlsx"
            data = [data]

            df = pd.DataFrame(data, columns=columns)
            df.to_excel(new_xlsx, index=False)
            break

    return new_xlsx
