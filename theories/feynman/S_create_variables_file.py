import pandas as pd
import numpy as np
import os

def create_variables_file(n,filename):
    header = np.array(["Filename", "Formula", "var1", "var2", "var3", "var4", "var5", "var6", "var7", "var8", "var9", "var10", "var11", "var12", "var13", "var14", "var15"])
    data = np.array([[filename, "", "x", "y", "z", "t", "w", "a", "b","c","d","e","f","g","alpha","beta","gamma"]])
    spaces = np.array(["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    df=pd.DataFrame(data = [np.concatenate((data[0][0:n+2],spaces[n:15]))], columns=header)
    print(os.getcwd())
    df.to_excel("results/mystery_world_solvedPart_newVars.xlsx")
    return "results/mystery_world_solvedPart_newVars.xlsx"
