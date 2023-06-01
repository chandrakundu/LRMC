## Training codes for LRPCA 
## Implemented by Jialin Liu @ Alibaba DAMO
## Date: Dec. 07, 2021

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
current_dir = os.getcwd()
import gc


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as lina
import time
import sys


## ================Preparations====================
device = torch.device('cuda:1')
datatype = torch.float32



## ================Parameters======================
n 				= 10000		# size of the problem
r 				= 5			# underlying rank
alpha 			= 0.1		# fraction of outliers
p 				= 0.02		# fraction of observed entries

d1 				= n		# size (num. of rows)
d2 				= n		# size (num. of columns)
step_initial 	= 0.55		# initial value of step size (eta in the paper)
ths_initial 	= 17/n		# initial value of thresholds (zeta in the paper)
maxIt 			= 32		# num. of layers you want to train

## =================Training Scripts======================
Nepoches_pre 	= 2
Nepoches_full 	= 2
nepoch          = 2
lr_fac 			= 1.0															# basic learning rate



## =============Generate RPCA problems=============
def generate_problem(r,d1,d2,alpha, p):
    U0_t 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d1)
    V0_t 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d2)
    idx 		= torch.randperm(d1*d2, device = device)
    idx 		= idx[:math.floor(alpha * d1*d2)]
    Y0_t 		= torch.mm(U0_t,V0_t.t())
    Y0_t 		= Y0_t.reshape(-1)
    s_range		= torch.mean(torch.abs(Y0_t))
    S0_t 		= torch.rand(len(idx), dtype = datatype, device = device)
    S0_t 		= s_range * (2.0 * S0_t - 1.0)
    Y0_t[idx] 	= Y0_t[idx] + S0_t
    Y0_t 		= Y0_t.reshape((d1,d2)) 
    omega_id 	= torch.randperm(d1*d2,device=device)[:int(d1*d2*(p))]
    omega 		= torch.zeros(d1*d2, dtype = datatype, device = device)
    omega.view(-1)[omega_id] = 1
    Y0_t 		= omega.view(d1,d2) * Y0_t     

    return U0_t, V0_t, Y0_t, omega.view(d1,d2)





## ===================LRPCA model===================
class MatNet(nn.Module):
    def __init__(self):
        super(type(self),self).__init__()
        self.ths_v 		= [nn.Parameter(Variable(torch.tensor(ths_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
        self.step 		= [nn.Parameter(Variable(torch.tensor(step_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
        self.ths_backup	= [torch.tensor(0.0, dtype=datatype, device = device) for t in range(maxIt)]

    # def thre(self, inputs, threshold):
    #     zero = torch.zeros([1, 1], dtype=datatype, device=device, requires_grad=False)
    #     out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, zero) 
    #     return out
    
    def thre(self, x, lambd):
        return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)

    def forward(self, Y0_t, r, U0_t, V0_t, p, omega, num_l):
        
        ## Initialization	

        S_t =  self.thre(Y0_t, self.ths_v[0])
        L, Sigma, R = torch.svd_lowrank((Y0_t-S_t)/p, q = r, niter = 4)
        Sigsqrt = torch.diag(torch.sqrt(Sigma))
        U_t = torch.mm(L, Sigsqrt)
        V_t = torch.mm(R, Sigsqrt) 		

        ## Main Loop in LRMC
        for t in range(1, num_l):
            X_t = torch.mm(U_t,V_t.t())
            YmUV = Y0_t - omega * X_t
            S_t =  self.thre(YmUV, self.ths_v[t])
            E_t = (YmUV - S_t) 
            Vkernel = torch.inverse(V_t.t() @ V_t)
            Ukernel = torch.inverse(U_t.t() @ U_t)
            Unew = U_t + self.step[t] * (torch.mm(E_t,V_t) @ Vkernel)
            Vnew = V_t + self.step[t] * (torch.mm(U_t.t(),E_t).t() @ Ukernel)
            U_t = Unew
            V_t = Vnew

        ## loss function in training
        loss = (torch.mm(U_t, V_t.t()) - torch.mm(U0_t, V0_t.t())).norm() 		
        return loss

    def InitializeThs(self, en_l):
        self.ths_v[en_l].data = torch.clone(self.ths_v[en_l-1].data * 0.66)
        
    def InitializeStep(self, en_l):        
        if en_l > 8:
            self.step[en_l].data = torch.tensor(1.0/p, dtype=datatype, device = device)
        else:
            eta_add = torch.tensor((1/p-1)*(step_initial/7))
            self.step[en_l].data = torch.clone(self.step[en_l-1].data + eta_add)
        
    def CheckNegative(self):
        isNegative = False;
        for t in range(maxIt):
            if(self.ths_v[t].data < 0):
                isNegative = True;
        if(isNegative):
            for t in range(maxIt):
                self.ths_v[t].data = torch.clone(self.ths_backup[t])
        else:
            for t in range(maxIt):
                self.ths_backup[t] = torch.clone(self.ths_v[t].data)
        return isNegative;

    def EnableSingleLayer(self,en_l):
        for t in range(maxIt): 
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False
        self.ths_v[en_l].requires_grad = True
        self.step[en_l].requires_grad = True

    def EnableLayers(self, num_l):
        for t in range(num_l): 
            self.ths_v[t].requires_grad = True
            self.step[t].requires_grad = True
        for t in range(num_l,maxIt): 
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False



net = MatNet()
optimizers = []
for i in range(maxIt):
    optimizer = optim.SGD({net.ths_v[i]},lr = lr_fac * ths_initial / 10000.0)	# optimizer for each layer
    optimizer.add_param_group({'params': [net.step[i]], 'lr': lr_fac /10000.0})	# learning rate for each layer
    optimizers.append(optimizer)
  
## =================Layerwise Training======================
start = time.time()
for stage in range(maxIt):														# in k-th stage, we train the k-th layer
    
    ## Pre-training: only train the k-th layer
    print('Layer ',stage,', Pre-training ======================')
    if(stage > 16):
        Nepoches_full = nepoch

    if(stage > 0):
        optimizers[stage].param_groups[0]['lr'] = net.ths_v[stage-1].data * lr_fac / 10000.0
    for epoch in range(Nepoches_pre):
        for i in range(maxIt):
            optimizers[i].zero_grad()
    
        U0_t,V0_t,Y0_t,omega = generate_problem(r,d1,d2,alpha,p)
        net.EnableSingleLayer(stage)
        if(stage > 0):
            net.InitializeThs(stage)
            net.InitializeStep(stage)
        loss = net(Y0_t, r, U0_t, V0_t,p,omega, stage+1)
        loss.backward()
        optimizers[stage].step()
    
        if(epoch % 10 == 0):
            if net.CheckNegative():
                print("Negative detected, restored")
                
        lr = optimizers[stage].param_groups[0]['lr']
        if epoch % 100 == 0:
            print("epoch: " + str(epoch), "\t loss: " + str(loss.item()))

        del U0_t,V0_t,Y0_t,omega
        gc.collect()
        torch.cuda.empty_cache()

    # Full-training: train 0~k th layers
    print('Layer ',stage,', Full-training =====================')
    if stage == 0:
        continue

    for epoch in range(Nepoches_full):
        for i in range(maxIt):
            optimizers[i].zero_grad()
    
        U0_t,V0_t,Y0_t,omega = generate_problem(r,d1,d2,alpha,p)
        net.EnableLayers(stage+1)
        loss = net(Y0_t, r, U0_t, V0_t,p,omega, stage+1)
        loss.backward()
        
        for i in range(stage+1):
            optimizers[i].step()

        if epoch % 20 == 0:
            print("epoch: " + str(epoch), "\t loss: " + str(loss.item()))

        del U0_t,V0_t,Y0_t,omega
        gc.collect()
        torch.cuda.empty_cache()
    
    
    
end = time.time()
training_time = end - start
print("Training end. Time: " + str(training_time))

## run test lrmc
from lrmc_generated_data_theoretical_model import generate_problem, lrmc
X_orig, Y, omega = generate_problem(r,n,n,alpha, p, device)
zeta = net.ths_v
eta = net.step
errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, maxIt, device)
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
plt.savefig(f'output/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_i{maxIt}.png')


## =====================Save model to .mat file ========================
result_ths 	= np.zeros((maxIt,))
result_stp1 = np.zeros((maxIt,))
result_stp2 = np.zeros((maxIt,))
for i in range(maxIt):
    result_ths[i] 	= net.ths_v[i].data.cpu().numpy()
    result_stp1[i] 	= net.step[i].data.cpu().numpy()

# spath = 'LRPCA_alpha'+str(alpha)+'.mat'
spath = f'saved_models/lrmc_n{n}_r{r}_a{alpha}_p{p}.mat'
sio.savemat(spath, {'zeta':result_ths, 'eta':result_stp1})

# path = f'log/lrmc_{n}_{r}_{alpha}.txt'
# with open(path, 'w') as f:
# 	f.write(f'Time: {training_time}, Loss: {loss}')










