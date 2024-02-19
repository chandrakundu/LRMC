import torch
from face_modeling.dataprocess import Face

# testing lrmc
from inc_face.lrmc.model import train, save
from inc_face.lrmc.networks import LRMCNet
from inc_face.lrpca.networks import LRPCANet
from inc_face.lrpca.model import train_lrpca
from inc_face.lrpca.model import save_lrpca


def main(r, p, max_iter, device):
    print("LRMC Start")
    zeta0 = 0.9
    eta0 = 1 / p
    data = Face(device, p)
    net = LRMCNet(r, p, max_iter, zeta0, eta0, device)
    nepoch_pre = 301
    nepoch = 301
    lr_inits = [1, 10, 100, 1000]
    lr_frac_zeta = 10000.0
    lr_frac_eta = 10000.0

    for lr_init in lr_inits:
        net = train(net, data, nepoch_pre, nepoch, lr_init, lr_frac_zeta, lr_frac_eta)
        save(net, "ext_lr_init2_" + str(lr_init))


def main_lrpca(r, max_iter, device):
    print("LRMC Start")
    zeta0 = 0.9
    eta0 = 1
    data = Face(device, p)
    net = LRPCANet(r, max_iter, zeta0, eta0, device)
    nepoch_pre = 101
    nepoch = 201
    lr_inits = [1, 10, 100, 1000]
    lr_frac_zeta = 10000.0
    lr_frac_eta = 10000.0

    for lr_init in lr_inits:
        net = train_lrpca(
            net, data, nepoch_pre, nepoch, lr_init, lr_frac_zeta, lr_frac_eta
        )
        save_lrpca(net, "ext_lr_init_" + str(lr_init))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p = 0.5
    r = 2

    max_iter = 10

    ps = [0.3]
    rs = [1]

    for p in ps:
        for r in rs:
            main(r, p, max_iter, device)
            # main_lrpca(r, max_iter, device)

    # main(r, p, max_iter, device)
    # main_lrpca(r, max_iter, device)
