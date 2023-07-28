import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import scipy.io as sio
from inc_lrpca.synthetic_data import DataGenerator


def soft_thres(inputs, threshold, device):
    out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], dtype=torch.float32, device=device) )
    return out

def lrpca(U,V,Y, r, zeta, eta, device):
    errors = []
    times = []
    thres_up = 1e3
    thres_low = 1e-8
    X_orig = torch.mm(U,V.t())

    start_time = time.time()
    S_t = soft_thres(Y, zeta[0], device)
    L, Sigma, R = torch.svd_lowrank(Y-S_t, q = r, niter = 4)
    Sigsqrt = torch.diag(torch.sqrt(Sigma))
    U_t = torch.mm(L, Sigsqrt)
    V_t = torch.mm(R, Sigsqrt)

    timer = time.time() - start_time

    error = torch.norm(X_orig- torch.mm(U_t,V_t.t()))/torch.norm(X_orig)
    errors.append(error)
    times.append(timer)
    print("Iteration: {}, Error: {}, Time: {}".format(0, error, timer))

    for i in range(1, len(zeta)):
        S_t = soft_thres(Y- torch.mm(U_t,V_t.t()), zeta[i], device)
        E_t = Y - S_t - torch.mm(U_t,V_t.t())
        Vkernel = torch.inverse(V_t.t() @ V_t)
        Ukernel = torch.inverse(U_t.t() @ U_t)
        Unew = U_t + eta[i] * (torch.mm(E_t,V_t) @ Vkernel)
        Vnew = V_t + eta[i] * (torch.mm(U_t.t(),E_t).t() @ Ukernel)
        U_t = Unew
        V_t = Vnew

        timer = time.time() - start_time

        error = torch.norm(X_orig- torch.mm(U_t,V_t.t()))/torch.norm(X_orig)
        errors.append(error)
        times.append(timer)
        print("Iteration: {}, Error: {}, Time: {}".format(i, error, timer))

    return torch.tensor(errors), torch.tensor(times)



if __name__ == "__main__":
    n1 = 1000
    n2 = 1000
    r = 5
    alpha = 0.1
    max_iter = 20
    device = torch.device('cuda:0')
    
    data = DataGenerator(n1, n2, r, alpha, device)
    U, V, Y = data.new()
    max_iter = 20
    max_iter = 32
    zeta = torch.tensor(17/n1, device=device)*0.1**torch.arange(0,max_iter,1, device=device)
    eta = torch.ones(max_iter, device=device) 

    errors, times = lrpca(U,V,Y, r, zeta, eta, device)

    plt.figure()
    # plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    # plt.savefig(f'output_lrmc/3/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_lrzeta{lr_frac_zeta}_lreta{lr_frac_eta}.png')
    # plt.close()
    plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    plt.savefig(f'output_lrpca/lrpca_n{n1}_r{r}_alpha{alpha}_i{max_iter}.png')
    
    


    

