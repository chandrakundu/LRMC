import torch
from ultrasound_data.dataprocess import PatchFile
# testing lrmc
from lrmc_generated_data_theoretical_model import lrmc_us
from inc_ultrasound.model import train,train_lrpca, save,save_lrpca
from inc_ultrasound.networks import LRMCNet, LRPCANet


def main(r, p, max_iter, device):
    print('LRMC Start')
    zeta0 = 0.1
    eta0 = 1/p
    data = PatchFile(device, p)
    net = LRMCNet(r, p, max_iter, zeta0, eta0, device)
    nepoch_pre = 301
    nepoch = 701
    lr_init = 1
    lr_frac_zeta = 10000.0
    lr_frac_eta = 10000.0
    net = train(net, data, nepoch_pre, nepoch, lr_init, lr_frac_zeta, lr_frac_eta)

    save(net, fname_prefix='lrmc2')

def main_lrpca(r,  max_iter, device):
    print('LRPCA Start')
    zeta0 = 0.1
    eta0 = 0.5
    data = PatchFile(device, p)
    net = LRPCANet(r, max_iter, zeta0, eta0, device)
    nepoch_pre = 301
    nepoch = 301
    lr_init = 1
    lr_frac_zeta = 5000.0
    lr_frac_eta = 10.0
    net = train_lrpca(net, data, nepoch_pre, nepoch, lr_init, lr_frac_zeta, lr_frac_eta)

    save_lrpca(net, fname_prefix='lrpca')

if __name__ == "__main__": 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = 0.30
    r = 1

    max_iter = 20
    
    # main(r, p, max_iter, device)
    main_lrpca(r, max_iter, device)
    # zeta = torch.max(torch.abs(X_orig))*0.66**torch.arange(0,max_iter,1, device=device)



    # p = 1 
    # r = 1 
    # max_iter = 20
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # zeta = torch.max(torch.abs(X_orig))*0.66**torch.arange(0,max_iter,1, device=device)
    # zeta = torch.tensor(0.1, device=device)*0.9**torch.arange(0,max_iter,1, device=device)
    # print(zeta)
    # eta = torch.ones(max_iter, device=device)/2

    # data = PatchFile(device, p)
    # L,S,Y, omega = data.new()

    # errors, times = lrmc_us(L, S, Y, r, omega, zeta, eta, max_iter, device)


