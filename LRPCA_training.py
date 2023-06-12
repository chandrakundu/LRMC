import torch
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from inc_lrpca.model import train, save
from inc_lrpca.network import LRPCANet
from inc_lrpca.synthetic_data import DataGenerator



def main(d1, d2, r, alpha, max_iter, device): 
    n=d1   
    zeta_init = 17/n
    eta_init = 0.55       

    net = LRPCANet(d1, d2, r, alpha, max_iter, zeta_init, eta_init, device)
    data = DataGenerator(net.d1, net.d2, net.r, net.alpha, net.device)

    nepoch_pre = 501
    nepoch = 1001
    lr_init = 1

    # print all the parameters
    # print(f'n: {n}, r: {r}, alpha: {alpha}, p: {p}, max_iter: {max_iter}, zeta_init: {zeta_init}, eta_init: {eta_init}, nepoch_pre: {nepoch_pre}, nepoch: {nepoch}, lr_init: {lr_init}' )
    

    net = train(net, data, nepoch_pre, nepoch, lr_init)
    save(net)


    return net

def run_main(n,r,alpha,max_iter,device):
    try:
        net = main(n, n, r, alpha, max_iter, device)
    except:
        with open('error_log.txt', 'a') as f:
            f.write(f'LRPCA: n: {n}, alpha: {alpha}, max_iter: {max_iter}\n')
        run_main(n,r,alpha,max_iter-2,device)


        


if __name__ == "__main__":
    # ns = [2000, 3000, 4000, 5000] 
    ns = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    # ns = [10000,3000]
    # ps = [0.0299, 0.0867]

    # ns = [9000, 4000]
    # ps = [0.0329, 0.0674]

    # ns = [8000, 5000]
    # ps = [0.0365, 0.0554]

    # ns = [7000, 6000]
    # ps = [0.0411, 0.0471]


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    r = 5
    # alpha = 0.1
    max_iter = 16

    for n in ns:
        if n == 4000 or n == 3000:
            alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4,0.45,0.5]
        else:
            alphas = [0.1]
        for alpha in alphas:
            run_main(n,r,alpha,max_iter,device)
    



    # lr_frac_zetas = [100,1000,10000]
    # lr_frac_etas = [10, 100]

    # for lr_frac_zeta in lr_frac_zetas:
    #     for lr_frac_eta in lr_frac_etas:
    #         net = main(n, n, r, alpha, p, max_iter,lr_frac_zeta, lr_frac_eta, device)
    
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
   


    