import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import scipy.io as sio

datatype = torch.float32

def generate_problem(r,d1,d2,alpha, p, device):

    # generate a low rank matrix
    U 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d1)
    V 		= torch.randn(d1,r,dtype = datatype, device = device)/math.sqrt(d2)
    X_orig 	= torch.mm(U,V.t())

    # generate observed matrix
    idx 		= torch.randperm(d1*d2, device = device)
    idx 		= idx[:math.floor(alpha * d1*d2)]
    Y           = torch.mm(U,V.t())
    Y           = Y.reshape(-1)

    # generate noise
    a	    	= torch.mean(torch.abs(Y))
    S    		= torch.rand(len(idx), dtype = datatype, device = device)
    S    		= 2*a* (S - 0.5)
    Y[idx] 		= Y[idx] + S
    Y 			= Y.reshape((d1,d2)) 
    omega_id 	= torch.randperm(d1*d2,device=device)[:int(d1*d2*(p))]
    omega 		= torch.zeros(d1*d2, dtype = datatype, device = device)
    omega.view(-1)[omega_id] = 1
    Y 		    = omega.view(d1,d2) * Y    

    return X_orig, Y, omega.view(d1,d2)

def lrmc(X_orig, Y, r, omega, zeta, eta, max_iter, device):
    # preparation 
    errors = []
    times = []
    thres_up = 1e3
    thres_low = 1e-8

    start_time = time.time()
    
    S0 = Y - soft_thres(Y, zeta[0], device)*eta[0]
    
    U0, Sigma0, V0 = torch.svd_lowrank(S0,r,niter=4)
    Sigma_Sqrt = torch.diag(torch.sqrt(Sigma0))
    L = torch.mm(U0,Sigma_Sqrt)
    R = torch.mm(V0, Sigma_Sqrt)
    timer = time.time() - start_time

    error = torch.norm(X_orig- torch.mm(L,R.t()))/torch.norm(X_orig)
    errors.append(error)
    times.append(timer)
    print("Iteration: {}, Error: {}, Time: {}".format(0, error, timer))

    for i in range(1, max_iter):
        start_time = time.time()
        diff = Y - omega*torch.mm(L,R.t())
        midterm = (soft_thres(diff, zeta[i], device) - diff)*eta[i]
        RRinv = torch.inverse(torch.mm(R.t(),R))
        LLinv = torch.inverse(torch.mm(L.t(),L))

        Lk = L - torch.mm(torch.mm(midterm, R), RRinv)
        Rk = R - torch.mm(torch.mm(midterm.t(), L), LLinv)
        L = Lk
        R = Rk
        timer += time.time() - start_time
        error = torch.norm(X_orig- torch.mm(L,R.t()))/torch.norm(X_orig)
        errors.append(error)
        times.append(timer)
        print("Iteration: {}, Error: {}, Time: {}".format(i, error, timer))
        if error < thres_low or error > thres_up:
            break

    return torch.tensor(errors), torch.tensor(times)

    

# def soft_thres(inputs, threshold, device):
# 		out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], device=device) )
# 		return out

def soft_thres(x, lambd, device):
    return torch.nn.functional.relu(x - lambd) - torch.nn.functional.relu(-x - lambd)

if __name__ == "__main__":
    r = 5
    n = 3000
    d1 = n
    d2 = n
    alpha = 0.1
    p = 0.07
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X_orig, Y, omega = generate_problem(r,d1,d2,alpha, p,device)
    
    
    max_iter = 32
    # zeta = torch.max(torch.abs(X_orig))*0.66**torch.arange(0,max_iter,1, device=device)
    zeta = torch.tensor(17/n, device=device)*0.66**torch.arange(0,max_iter,1, device=device)
    eta = torch.ones(max_iter, device=device)/p 
    eta[:8] = torch.linspace(1, 1 / p, 8) * 0.55
   

    errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, max_iter, device)
    
    plt.figure()
    # mark with o
    plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    plt.savefig('output/lrmc_generated_data_theoretical_model.png')