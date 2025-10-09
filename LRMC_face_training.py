import torch
from face_modeling.dataprocess import Face
import os
import click
import numpy as np

# testing lrmc
from inc_face.lrmc.model import train_lrmc, save_lrmc
from inc_face.lrmc.networks import LRMCNet
from inc_face.lrpca.networks import LRPCANet
from inc_face.lrpca.model import train_lrpca
from inc_face.lrpca.model import save_lrpca


def error_handling(e, dirname, r, p, lr_zeta, lr_eta):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created directory {dirname}. Now saving model...")
    with open(f"{dirname}/error_log.txt", "a") as f:
        f.write(f"LRMC: r: {r}, p: {p}, lr_zeta: {lr_zeta}, lr_eta: {lr_eta}\n")
        f.write(f"Error: {e}\n")
        f.write("-----------------------------------\n")


def main_lrmc(
    r,
    p,
    max_iter,
    zeta_inits,
    eta_inits,
    lr_zeta,
    lr_eta,
    device,
    dir="output",
    fname_ext="",
):
    print("LRMC Started... ")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    data = Face(device, p)

    net = LRMCNet(r, p, max_iter, zeta_inits, eta_inits, device)
    nepoch_pre = 501
    nepoch = 501

    print(
        f"device: {device}, r: {r}, p: {p}, max_iter: {max_iter}, lr_zeta: {lr_zeta}, lr_eta: {lr_eta}"
    )

    net = train_lrmc(net, data, nepoch_pre, nepoch, lr_zeta, lr_eta)
    save_lrmc(net, fname_ext=fname_ext, dirname=dir)

    del net, data
    torch.cuda.empty_cache()


def main_lrpca(
    r, max_iter, zeta0, eta0, lr_zeta, lr_eta, device, dir="output", fname_ext=""
):
    print("LRPCA Started ....")
    p = 1
    data = Face(device, p)
    net = LRPCANet(r, max_iter, zeta0, eta0, device)
    nepoch_pre = 501
    nepoch = 501

    net = train_lrpca(net, data, nepoch_pre, nepoch, lr_zeta, lr_eta)
    save_lrpca(net, fname_ext=fname_ext, dirname=dir)

    del net, data
    torch.cuda.empty_cache()


@click.command()
@click.option("-mt", "--model_type", default=0, help="LRMC(0) or LRPCA(0)")
@click.option("-r", "--r", default=1, help="Rank")
@click.option("-p", "--p", default=0.3, help="Probability")
@click.option("-miter", "--max_iter", default=40, help="Max Iteration")
@click.option("-e", "--eta0", default=1, help="Eta Inits or mult")
@click.option("-z", "--zeta0", default=0.7, help="Zeta Inits or mult")
@click.option("-d", "--device", default=0, help="GPU device")
@click.option("-lrz", "--lr_zeta", default=0.001, help="Learning rate for Zeta")
@click.option("-lre", "--lr_eta", default=0.001, help="Learning rate for Eta")
def cli(model_type, r, p, max_iter, zeta0, eta0, lr_zeta, lr_eta, device):

    r = 1
    if model_type == 0:
        dirname = f"output/lrmc/sept24/face/{lr_zeta}"
        eta_inits = np.repeat(1 / p, max_iter)
        zeta_inits = [240 * (zeta0) ** i for i in range(max_iter)]
        main_lrmc(
            r, p, max_iter, zeta_inits, eta_inits, lr_zeta, lr_eta, device, dir=dirname
        )
    else:
        dirname = f"output/lrpca/sept24/face/{lr_zeta}"
        zeta0 = zeta0 * 240
        main_lrpca(
            r,
            max_iter,
            zeta0,
            eta0,
            lr_zeta,
            lr_eta,
            device,
            dir=dirname,
        )


if __name__ == "__main__":
    cli()
