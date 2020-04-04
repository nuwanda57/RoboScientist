from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from matplotlib import pyplot as plt
import torch.utils.data as utils
import time

bs = 2048
wd = 1e-2

is_cuda = torch.cuda.is_available()

class MultDataset(data.Dataset):
    def __init__(self, factors, product):
        'Initialization'
        self.factors = factors
        self.product = product

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.product)

    def __getitem__(self, index):
        # Load data and get label
        x = self.factors[index]
        y = self.product[index]

        return x, y

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def NN_train(pathdir, filename, epochs=-1):
    try:
        os.mkdir("results/NN_trained_models/")
    except:
        pass

    try:
        os.mkdir("results/NN_trained_models/models/")
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))

        if epochs==-1:
            epochs = 25*n_variables
        else:
            epochs = epochs//4


        if n_variables==0:
            print("Solved! ", variables[0])
            return 0
        elif n_variables==1:
            variables = np.reshape(variables,(len(variables),1))
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables[0:int(5*len(variables)/6)])
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent[0:int(5*len(f_dependent)/6)])
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        factors_val = torch.from_numpy(variables[int(5*len(variables)/6):int(len(variables))])
        if is_cuda:
            factors_val = factors_val.cuda()
        else:
            factors_val = factors_val
        factors_val = factors_val.float()

        product_val = torch.from_numpy(f_dependent[int(5*len(variables)/6):int(len(variables))])      
        if is_cuda:
            product_val = product_val.cuda()
        else:
            product_val = product_val
        product_val = product_val.float()

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

        my_dataset = utils.TensorDataset(factors,product) # create your datset
        my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=True) # create your dataloader

        if is_cuda:
            model_feynman = SimpleNet(n_variables).cuda()
        else:
            model_feynman = SimpleNet(n_variables)

        lrs = 1e-3
        for i_i in range(4):
            optimizer_feynman = optim.Adam(model_feynman.parameters(), lr = lrs)
            for epoch in range(epochs):
                model_feynman.train()
                for i, data in enumerate(my_dataloader):
                    optimizer_feynman.zero_grad()
                
                    if is_cuda:
                        fct = data[0].float().cuda()
                        prd = data[1].float().cuda()
                    else:
                        fct = data[0].float()
                        prd = data[1].float()
                    
                    loss = rmse_loss(model_feynman(fct),prd)
                    loss.backward()
                    optimizer_feynman.step()
            print(loss)
            lrs = lrs/10

        torch.save(model_feynman.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")

        return 1

    except:
        print("Error in file: %s" %filename)
        return 0



