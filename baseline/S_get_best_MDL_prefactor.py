import sys
import numpy as np

denominator = np.loadtxt("results/RMS_file_value.txt",usecols=(0,))

filename = "results.dat"

d1 = np.loadtxt(filename,usecols=(0,))
d2 = np.loadtxt(filename,dtype="str",usecols=(1,))
d3 = np.loadtxt(filename,usecols=(2,))
d4 = np.loadtxt(filename,usecols=(3,))
d5 = np.loadtxt(filename,usecols=(4,))
d6 = np.loadtxt(filename,usecols=(5,))
d7 = np.loadtxt(filename,usecols=(6,))

#i = np.argmin(d7) - for long expressions
i = np.argmin(d6)

d1 = np.append(d1,d1[i])
d1 = d1/denominator
d2 = np.append(d2,d2[i])
d3 = np.append(d3,d3[i])
d4 = np.append(d4,d4[i])
d4 = d4/denominator
d5 = np.append(d5,d5[i])
d6 = np.append(d6,d6[i])
d7 = np.append(d7,d7[i])

data = np.column_stack((d1,d2))
data = np.column_stack((data,d3))
data = np.column_stack((data,d4))
data = np.column_stack((data,d5))
data = np.column_stack((data,d6))
data = np.column_stack((data,d7))


f= open(filename,"w+")

for j in range(len(d1)):
    f.write(str(d1[j]))
    f.write(" ")
    f.write(d2[j])
    f.write(" ")
    f.write(str(d3[j]))
    f.write(" ")
    f.write(str(d4[j]))
    f.write(" ")
    f.write(str(d5[j]))
    f.write(" ")
    f.write(str(d6[j]))
    f.write(" ")
    f.write(str(d7[j]))
    f.write("\n")
