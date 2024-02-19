import torch
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from inc_virat.model import train, save
from inc_virat.networks import LRMCNet
from virat_data.dataprocess import VideoFile

# testing lrmc
from inc_virat.lrmc_generated_data_theoretical_model import lrmc


def main(d1, d2, r, alpha, p, max_iter, device):
    n = d1
    zeta_init = 255
    eta_init = 1 / p

    net = LRMCNet(d1, d2, r, alpha, p, max_iter, zeta_init, eta_init, device)
    data = VideoFile(device, p, max_frame=1000)

    nepoch_pre = 501
    nepoch = 501
    lr_init = 10

    # print all the parameters
    print(
        f"n: {n}, r: {r}, alpha: {alpha}, p: {p}, max_iter: {max_iter}, zeta_init: {zeta_init}, eta_init: {eta_init}, nepoch_pre: {nepoch_pre}, nepoch: {nepoch}, lr_init: {lr_init}"
    )

    net = train(net, data, nepoch_pre, nepoch, lr_init)
    save(net, "exp_lr_10_1_it10", "virat_saved_model_jan24")

    X_orig, Y, omega = data.new()
    zeta = net.zeta
    eta = net.eta

    with torch.no_grad():
        errors, times = lrmc(X_orig, Y, r, omega, zeta, eta, max_iter, device)

    plt.figure()
    plt.semilogy(times.detach().numpy(), errors.detach().numpy(), "o-")
    plt.savefig(
        f"virat_saved_model_jan24/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_i{max_iter}_exp_lr_10.png"
    )

    return net


if __name__ == "__main__":
    p = 0.1
    r = 2
    width = 320
    height = 180
    d1 = width
    d2 = height
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # data = VideoFile(device, p, max_frame=1000)

    r = 2
    alpha = 0.1
    max_iter = 10

    net = main(d1, d2, r, alpha, p, max_iter, device)
