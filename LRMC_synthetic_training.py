import torch
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from inc.model import train, save
from inc.networks import LRMCNet
from inc.synthetic_data import DataGenerator

# testing lrmc
from lrmc_generated_data_theoretical_model import generate_problem, lrmc



def main(d1, d2, r, alpha, p, max_iter, lr_frac_zeta, lr_frac_eta, device): 
    n=d1   
    zeta_init = 17/n
    eta_init = 0.55       

    net = LRMCNet(d1, d2, r, alpha, p, max_iter, zeta_init, eta_init, device)
    data = DataGenerator(net.d1, net.d2, net.r, net.alpha, net.p, net.device)

    nepoch_pre = 301
    nepoch = 1001
    lr_init = 1

    # print all the parameters
    print(f'device: {device},n: {n}, r: {r}, alpha: {alpha}, p: {p}, max_iter: {max_iter}, zeta_init: {zeta_init}, eta_init: {eta_init}, nepoch_pre: {nepoch_pre}, nepoch: {nepoch}, lr_init: {lr_init}' )
    # print(f'lr_frac_zeta: {lr_frac_zeta}, lr_frac_eta: {lr_frac_eta}, device: {device}' )

    net = train(net, data, nepoch_pre, nepoch, lr_init, lr_frac_zeta, lr_frac_eta)
    # ext = f'lrzeta{lr_frac_zeta}_lreta{lr_frac_eta}'
    save(net)

    U, V, Y, omega = data.new()
    X_orig = U @ V.T
    zeta = net.zeta
    eta = net.eta

    errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, max_iter, device)
    plt.figure()
    plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    plt.savefig(f'output_lrmc/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_lrzeta{lr_frac_zeta}_lreta{lr_frac_eta}.png')

    return net


if __name__ == "__main__":
    # ns = [1000, 2000, 3000, 4000, 5000] 
    # ns = [1500, 2500, 3500, 4500]

    # ps = [0.2, 0.1, 0.07, 0.055, 0.045]
    n = 4000
    p = 0.08
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    r = 5
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] 
    max_iter = 32

    lr_frac_zeta = 1000
    lr_frac_eta = 1000

    for alpha in alphas:
        net = main(n, n, r, alpha, p, max_iter, lr_frac_zeta, lr_frac_eta, device)



    # for n in ns:
    #     net = main(n, n, r, alpha, p, max_iter, lr_frac_zeta, lr_frac_eta, device)

    

    
    
    # params1 = {
    #     'n': 10000,
    #     'p': 0.02,
    # }
    # params2 = {
    #     'n': 15000,
    #     'p': 0.015,
    # }
    # params3 = {
    #     'n': 20000,
    #     'p': 0.01,
    # }
    # params = [params1, params2, params3]

    # for param in params:
    #     n = param['n']
    #     p = param['p']
    #     net = main(n, n, r, alpha, p, max_iter, device)

    # testing lrmc
    # X_orig, Y, omega = generate_problem(r,n,n,alpha, p)
    # zeta = net.zeta
    # eta = net.eta

    # errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, max_iter)
    # plt.figure()
    # plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    # plt.savefig(f'output/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_i{max_iter}.png')
   


    