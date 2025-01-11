import torch
import numpy as np

from dataset import SyntheticData
from networks import LRMCNet
from trainer import train
from utils import save


def main():
    n = 3000  # number of rows
    r = 5  # rank of the matrix
    alpha = 0.1  # fraction of corrupted entries
    p = 1  # fraction of observed entries
    max_iter = 16  # number of iterations
    nepoch = 1000  # number of training epochs
    lr_zeta = 0.1  # learning rate for zeta
    lr_eta = 0.1  # learning rate for eta
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device

    # synthetic data generator
    data = SyntheticData(n, n, r, alpha, p, device)

    # when p < 1. initialize zeta with the following
    # zeta_inits = [17 / n] * max_iter
    # eta_inits = [1 / p] * max_iter

    # when p = 1, initialize zeta with the following
    zeta_inits = [1e-3] * max_iter
    eta_inits = [0.5] * max_iter

    # initialize the network
    net = LRMCNet(n, n, r, alpha, p, max_iter, zeta_inits, eta_inits, device)

    # train the network
    net = train(net, data, nepoch, lr_zeta, lr_eta)

    # save the trained model
    save(net, f"trained_model/lrmc_n{n}_r{r}_alpha{alpha}_p{p}.mat")


if __name__ == "__main__":
    print("LRMC training started ...")
    main()
