import numpy as np
import os
import subprocess
import sys
import importlib
from sympy.parsing.sympy_parser import parse_expr
from baseline.S_readable_formulas_from_bruteForce_v1 import readable_formulas_from_bruteForce_v1
from baseline.S_readable_formulas_from_bruteForce_v2 import readable_formulas_from_bruteForce_v2


# sep_type = 3 for add and 2 for mult and 1 for normal
def brute_force(pathdir,filename,methods_tried,method_name,BF_try_time,BF_ops_file_type,sep_type,use_MDL, check_prefactor):
    #np.savetxt("bruteformulas.txt",[])
    try_time = BF_try_time
    try_time_prefactor = BF_try_time
    file_type = BF_ops_file_type
    methods_tried = methods_tried + [method_name]
    try:
        os.remove("results.dat")
    except:
        pass
    print("LOOK FOR FORMULA")
    if sep_type==2:
        print("TYPE 2: ",pathdir,filename)
        if use_MDL:
            print('Calling %s' % "./brute_force_oneFile_use_MDL_v2.scr")
            subprocess.call(["./brute_force_oneFile_use_MDL_v2.scr", file_type, "%s" %try_time, pathdir+filename])
        else:
            print('Calling %s' % "./brute_force_oneFile_v2.scr")
            subprocess.call(["./brute_force_oneFile_v2.scr", file_type, "%s" %try_time, pathdir+filename])
        readable_formulas_from_bruteForce_v1()
    if sep_type==3:
        print("TYPE 3: ",pathdir,filename)
        if use_MDL:
            subprocess.call(["./brute_force_oneFile_use_MDL_v3.scr", file_type, "%s" %try_time, pathdir+filename])
        else:
            subprocess.call(["./brute_force_oneFile_v3.scr", file_type, "%s" %try_time, pathdir+filename])
        readable_formulas_from_bruteForce_v1()
    if os.stat("bruteformulas.txt").st_size != 0:
        BF_formula = np.loadtxt("bruteformulas.txt", dtype=bytes, delimiter="\n").astype(str)
        BF_formula = str(BF_formula)
        BF_formula = BF_formula.replace("[", "(")
        BF_formula = BF_formula.replace("]", ")")
        BF_formula = BF_formula.replace("^", "**")
        print("TEST: ",BF_formula)
        # use this for NN separability
        BF_formula_noPrefactor = BF_formula
        BF_formula_noPrefactor = parse_expr(BF_formula_noPrefactor)
        # get the prefactor 
        if (sep_type==2 or sep_type==3) and check_prefactor==1:
            try:
                os.remove("results.dat")
            except:
                pass
            print("LOOK FOR PREFACTOR")
            BF_prefactor_unsolved = np.loadtxt("bruteprefactor.txt") # use this if the BF doesn't find an expression for the prefactor
            print("BF_prefactor_unsolved: ", BF_prefactor_unsolved)
            subprocess.call(["./brute_force_oneFile.scr", file_type, "%s" %try_time_prefactor, "bruteprefactor.txt"])
            readable_formulas_from_bruteForce_v2()
            if os.stat("bruteformulas.txt").st_size == 0:
                BF_prefactor = BF_prefactor_unsolved
                BF_prefactor = str(BF_prefactor)
            else:
                BF_prefactor = np.loadtxt("bruteformulas.txt", dtype=bytes, delimiter="\n").astype(str)
                BF_prefactor = str(BF_prefactor)
                print("Prefactor: ", BF_prefactor)
                BF_prefactor = BF_prefactor.replace("[", "(")
                BF_prefactor = BF_prefactor.replace("]", ")")
                BF_prefactor = BF_prefactor.replace("^", "**")
            # combined the formula with the prefactor 
            if sep_type==2:
                BF_formula = parse_expr(BF_formula)*parse_expr(BF_prefactor)
            if sep_type==3:
                BF_formula = parse_expr(BF_formula)+parse_expr(BF_prefactor)
        methods_tried = methods_tried + ["solved"]
        print("OVERALL FORMULA: ",BF_formula)
        if sep_type==1:
            return (parse_expr(BF_formula), methods_tried)
        else:
            return (BF_formula, methods_tried)
    else:
        return(0,methods_tried)


 
