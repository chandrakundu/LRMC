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
    ext = f'lrzeta{lr_frac_zeta}_lreta{lr_frac_eta}'
    save(net, ext)

    U, V, Y, omega = data.new()
    X_orig = U @ V.T
    zeta = net.zeta
    eta = net.eta

    errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, max_iter, device)
    plt.figure()
    plt.semilogy(times.detach().numpy(), errors.detach().numpy(), 'o-')
    plt.savefig(f'output_lrmc/3/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_lrzeta{lr_frac_zeta}_lreta{lr_frac_eta}.png')
    plt.close()
    return net


if __name__ == "__main__":
    # ns = [1000, 2000, 3000, 4000, 5000] 
    # ns = [1500, 2500, 3500, 4500]

    # ps = [0.2, 0.1, 0.07, 0.055, 0.045]

    # ns = [3000, 10000]
    # ps = [0.0867, 0.0299]

    # ns = [4000, 9000]
    # ps = [0.0674, 0.0329]

    # ns = [8000, 5000]
    # ps = [0.0365, 0.0554]

    # ns = [7000, 6000]
    # ps = [0.0411, 0.0471]

    # formula for ps: 1.3*n*r^2*log(n)/n^2

    # ns = [3000, 5000, 7000]
    # ps = [0.0867, 0.0554, 0.0411]

    # ns = [4000, 6000, 8000]
    # ps = [0.0674, 0.0471, 0.0365]

    # ns = [3000, 4000, 5000, 6000, 7000, 8000]
    # ps = [0.0867, 0.0674, 0.0554, 0.0471, 0.0411, 0.0365]    

    # ns = [3500, 4500, 5500, 6500, 7500]
    # ps = [0.0758, 0.0608, 0.0509, 0.0439, 0.0387]

    # ns = [9500, 10000]
    # ps = [0.0313, 0.0299]

    ns = [3000, 4000, 4500, 10000]
    ps = [0.0867, 0.0674, 0.0608, 0.0299]

    params = dict(zip(ns, ps))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    r = 5
    # alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] 
    
    max_iter = 64

    lr_frac_zetas = [10000, 1000]
    lr_frac_etas = [10000, 1000]

    for lr_frac_zeta in lr_frac_zetas:
        for lr_frac_eta in lr_frac_etas:            
            for n, p in params.items():
                if n == 4000 or n == 3000:
                    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4,0.45,0.5, 0.55, 0.6]
                else:
                    alphas = [0.1]
                
               

                if n > 9500:
                    max_iter = 16                    
                elif n > 4000:
                    max_iter = 32
                else:   
                    max_iter = 64

                for alpha in alphas:
                    try:
                        net = main(n, n, r, alpha, p, max_iter, lr_frac_zeta, lr_frac_eta, device)
                    except:
                        with open('error_log.txt', 'a') as f:
                            f.write(f'LRMC: n: {n}, p: {p}, alpha: {alpha}, lr_frac_zeta: {lr_frac_zeta}, lr_frac_eta: {lr_frac_eta}\n')
                        continue


    



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
   


    