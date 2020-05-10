import numpy as np
import pandas as pd


def readable_formulas_from_bruteForce_v1():
    error_ths = np.loadtxt("BF_error_threshold_file.txt")

    bf_output = pd.read_csv("brute_results.csv")

    variables = ["0", "1", "a", "b", "c", "d", "e", "f", "g", "h", "P"]
    operations_1 = [">", "<", "~", "\\", "L", "E", "S", "C", "A", "N", "T", "R"]
    operations_2 = ["+", "*", "-", "/"]

    error_fit = float(bf_output.columns[1])
    stack = np.array([])
    prefactor = float(bf_output.columns[4])

    if error_ths > error_fit:
        for i in (bf_output.columns[3]):
            if i in variables:
                if i == "P":
                    stack = np.append(stack, "Pi")
                else:
                    stack = np.append(stack, i)
            elif i in operations_2:
                a1 = stack[-1]
                a2 = stack[-2]
                stack = np.delete(stack, -1)
                stack = np.delete(stack, -1)
                a = "(" + a2 + i + a1 + ")"
                stack = np.append(stack, a)
            elif i in operations_1:
                a = stack[-1]
                stack = np.delete(stack, -1)
                if i == ">":
                    a = "(" + a + "+1)"
                    stack = np.append(stack, a)
                if i == "<":
                    a = "(" + a + "-1)"
                    stack = np.append(stack, a)
                if i == "~":
                    a = "(-" + a + ")"
                    stack = np.append(stack, a)
                if i == "\\":
                    a = "(" + a + ")^(-1)"
                    stack = np.append(stack, a)
                if i == "L":
                    a = "log(" + a + ")"
                    stack = np.append(stack, a)
                if i == "E":
                    a = "exp(" + a + ")"
                    stack = np.append(stack, a)
                if i == "S":
                    a = "sin(" + a + ")"
                    stack = np.append(stack, a)
                if i == "C":
                    a = "cos(" + a + ")"
                    stack = np.append(stack, a)
                if i == "A":
                    a = "abs(" + a + ")"
                    stack = np.append(stack, a)
                if i == "N":
                    a = "asin(" + a + ")"
                    stack = np.append(stack, a)
                if i == "T":
                    a = "atan(" + a + ")"
                    stack = np.append(stack, a)
                if i == "R":
                    a = "sqrt(" + a + ")"
                    stack = np.append(stack, a)
        np.savetxt("bruteformulas.txt", [stack[0]], fmt="%s")
        np.savetxt("bruteprefactor.txt", [prefactor])
    else:
        np.savetxt("bruteformulas.txt", [])
        np.savetxt("bruteprefactor.txt", [prefactor])
