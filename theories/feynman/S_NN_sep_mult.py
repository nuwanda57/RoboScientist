from __future__ import print_function
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from itertools import combinations
import time

is_cuda = torch.cuda.is_available()

class SimpleNet(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.linear1 = nn.Linear(ni, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64,64)
        self.linear6 = nn.Linear(64,64)
        self.linear7 = nn.Linear(64,1)
    
    def forward(self, x):
        x = F.softplus(self.linear1(x))
        x = F.softplus(self.linear2(x))
        x = F.softplus(self.linear3(x))
        x = F.softplus(self.linear4(x))
        x = F.softplus(self.linear5(x))
        x = F.softplus(self.linear6(x))
        x = self.linear7(x)
        return x

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def NN_sep_mult(pathdir, filename, err_sep_mult_factor=-1):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (0,0,0,0)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                variables = np.column_stack((variables,v))
        

        f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables)
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model_feynman = SimpleNet(n_variables).cuda()
        else:
            model_feynman = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        # make some variables at the time equal to the mean of factors

        facts = factors
        models_one = []
        models_rest = []

        with torch.no_grad():
            if rmse_loss(model(factors),product)>0.01:
                return (0,0,0,0)

            # get the error threshold for separability
            if err_sep_mult_factor==-1:
                error_threshold = 10*rmse_loss(model(factors),product)
            else:
                error_threshold = err_sep_mult_factor*rmse_loss(model(factors),product)
            print("ERROR CHECK: ",error_threshold)

            fact_vary = facts.clone()
            for k in range(len(facts[0])):
                fact_vary[:,k] = torch.full((len(facts),),torch.mean(factors[:,k]))

            # loop through all indices combinations
            var_indices_list = np.arange(0,n_variables,1)
            for i in range(1,n_variables):
                c = combinations(var_indices_list, i)
                for j in c:
                    print(j)
                    fact_vary_one = facts.clone()
                    fact_vary_rest = facts.clone()
                    rest_indx = list(filter(lambda x: x not in j, var_indices_list))
                    for t1 in rest_indx:
                        fact_vary_one[:,t1] = torch.full((len(facts),),torch.mean(factors[:,t1]))
                    for t2 in j:
                        fact_vary_rest[:,t2] = torch.full((len(facts),),torch.mean(factors[:,t2]))
                    
                    # check if the equation is separable
                    pd = model(fact_vary_one)*model(fact_vary_rest)
                    per_error = torch.sqrt(torch.mean((model(facts)-pd/model(fact_vary))**2))/torch.sqrt(torch.mean(model(facts)**2))
                    print(per_error)
                    if per_error<error_threshold:
                        str1 = filename+"-mult_a"
                        str2 = filename+"-mult_b"
                        # save the first half
                        data_sep_1 = variables
                        data_sep_1 = np.delete(data_sep_1,rest_indx,axis=1)
                        data_sep_1 = np.column_stack((data_sep_1,model(fact_vary_one).cpu()))
                        # save the second half  
                        data_sep_2 = variables
                        data_sep_2 = np.delete(data_sep_2,j,axis=1)
                        data_sep_2 = np.column_stack((data_sep_2,model(fact_vary_rest).cpu()/model(fact_vary).cpu()))
                        try:
                            os.mkdir("results/separable_mult/")
                        except:
                            pass
                        np.savetxt("results/separable_mult/"+str1,data_sep_1)
                        np.savetxt("results/separable_mult/"+str2,data_sep_2)
                        # if it is separable, return the 2 new files created and the index of the column with the separable variable
                        return (str1,str2, j, rest_indx)

    except Exception as e:
        print(e)
        return (0,0,0,0)

    return (0,0,0,0)
    

