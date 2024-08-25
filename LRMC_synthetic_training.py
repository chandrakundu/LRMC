import torch
import matplotlib.pyplot as plt
import os
import click
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from inc.model import train, save
from inc.networks import LRMCNet
from inc.synthetic_data import DataGenerator

# testing lrmc
from lrmc_generated_data_theoretical_model import generate_problem, lrmc


def error_handling(e, dirname, n, p, alpha, lr_zeta, lr_eta):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created directory {dirname}. Now saving model...")
    with open(f"{dirname}/error_log.txt", "a") as f:
        f.write(
            f"LRMC: n: {n}, p: {p}, alpha: {alpha}, lr_zeta: {lr_zeta}, lr_eta: {lr_eta}\n"
        )
        f.write(f"Error: {e}\n")
        f.write("-----------------------------------\n")


def main(
    d1,
    d2,
    r,
    alpha,
    p,
    max_iter,
    eta_inits,
    zeta_inits,
    lr_zeta,
    lr_eta,
    device,
    dir="output",
    fname_ext="",
):
    n = d1
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    net = LRMCNet(d1, d2, r, alpha, p, max_iter, zeta_inits, eta_inits, device)
    data = DataGenerator(net.d1, net.d2, net.r, net.alpha, net.p, net.device)

    nepoch_pre = 501
    nepoch = 501
    lr_init = 1

    # print all the parameters
    print(
        f"device: {device},n: {n}, r: {r}, alpha: {alpha}, p: {p}, max_iter: {max_iter}, lr_zeta: {lr_zeta}, lr_eta: {lr_eta}"
    )
    # print(f'lr_frac_zeta: {lr_frac_zeta}, lr_frac_eta: {lr_frac_eta}, device: {device}' )

    net = train(net, data, nepoch_pre, nepoch, lr_zeta, lr_eta)
    save(net, fname_ext=fname_ext, dirname=dir)

    U, V, Y, omega = data.new()

    try:
        with torch.no_grad():
            errors, times = lrmc(U, V, Y, r, omega, net.zeta, net.eta, max_iter, device)
        dirname = f"{dir}/images"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"Created directory {dirname}. Now saving model...")
        plt.figure()
        plt.semilogy(times.detach().numpy(), errors.detach().numpy(), "o-")
        plt.savefig(
            f"{dirname}/lrmc_n{n}_r{r}_alpha{alpha}_p{p}_lrz{lr_zeta}_lre{lr_eta}.png"
        )
        plt.close()
    except Exception as e:
        with open(f"{dirname}/error_log.txt", "a") as f:
            f.write(
                f"LRMC Plot Error: n: {n}, p: {p}, alpha: {alpha}, lr_frac_zeta: {lr_zeta}, lr_frac_eta: {lr_eta}\n"
            )
            f.write(f"Error: {e}\n")
            f.write("-----------------------------------\n")
    del net
    del U, V, Y, omega
    torch.cuda.empty_cache()


@click.command()
@click.option("-d", "--device", default=0, help="GPU device")
@click.option("-a", "--alpha", default=0.1, help="alpha")
@click.option("-p", "--p", default=0.1, help="p")
@click.option("-em", "--eta_mult", default=0.85, help="eta multiplier")
@click.option("-zm", "--zeta_mult", default=0.85, help="zeta multiplier")
@click.option("-miter", "--max_iter", default=20, help="max iterations")
def cli(device, alpha, p, eta_mult, zeta_mult, max_iter):
    n = 3000
    r = 5
    dirname = "output/lrmc/july24/different_p"
    eta_inits = np.repeat(1 / p, max_iter)
    eta_inits[1:9] = np.linspace(1, 1 / p, 8) * eta_mult
    zeta_inits = [(17 / n) * (zeta_mult) ** i for i in range(max_iter)]

    lr_etas = [0.001]
    lr_zetas = [0.001]

    lrs = zip(lr_zetas, lr_etas)

    for lr_zeta, lr_eta in lrs:
        try:
            # fname_ext = f"lrz{lr_zeta}_lre{lr_eta}"
            fname_ext = ""
            main(
                n,
                n,
                r,
                alpha,
                p,
                max_iter,
                eta_inits,
                zeta_inits,
                lr_zeta,
                lr_eta,
                device,
                dirname,
                fname_ext,
            )
        except Exception as e:
            error_handling(e, dirname, n, p, alpha, lr_zeta, lr_eta)
            continue


if __name__ == "__main__":
    cli()
