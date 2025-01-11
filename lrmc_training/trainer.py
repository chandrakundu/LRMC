import torch
import time
import gc


def train(net, data, nepoch, lr_zeta, lr_eta):
    """Training of LRMC
    Args:
        net: The LRMC network model to be trained
        data: Data loader providing training samples
        nepoch (int): Number of epochs for each training phase
        lr_zeta (float): Learning rate for zeta parameters
        lr_eta (float): Learning rate for eta parameters
    Returns:
        net: The trained network model
    """

    optimizers = []

    for t in range(net.max_iter):
        optimizer = torch.optim.SGD({net.zeta[t]}, lr=lr_zeta / 5000.0)
        optimizer.add_param_group({"params": net.eta[t], "lr": lr_eta})
        optimizers.append(optimizer)

    start = time.time()
    for stage in range(net.max_iter):
        print(f"Layer {stage} Pre-training ========================")

        if stage > 0:
            optimizers[stage].param_groups[0]["lr"] = net.zeta[stage - 1].data / 5000.0

        for epoch in range(nepoch):

            for i in range(net.max_iter):
                optimizers[i].zero_grad()

            net.enable_single_layer(stage)
            if stage > 0:
                net.initalize_zeta(stage)
            #     net.initalize_eta(stage)

            U, V, Y, omega = data.new()
            loss = net(U, V, Y, omega, stage + 1)

            loss.backward()
            optimizers[stage].step()

            if epoch % 10 == 0:
                if net.check_negative():
                    print("Negative zeta detected and reset to previous value")

            if (epoch + 1) % 100 == 0 and epoch > 0:
                print(f"Pretrain: Layer {stage} Epoch {epoch+1}: Loss {loss.item()}")

            # del U, V, Y, omega, loss
            gc.collect()
            torch.cuda.empty_cache()

        if stage == 0:
            continue

        print(f"Layer {stage} Full Training ========================")
        for epoch in range(nepoch):

            for i in range(net.max_iter):
                optimizers[i].zero_grad()

            net.enable_layers(stage + 1)

            U, V, Y, omega = data.new()
            loss = net(U, V, Y, omega, stage + 1)

            loss.backward()

            for i in range(stage + 1):
                optimizers[i].step()

            if (epoch + 1) % 100 == 0 and epoch > 0:
                print(f"Fulltrain: Layer {stage} Epoch {epoch+1}: Loss {loss.item()}")

            # del U, V, Y, omega, loss
            gc.collect()
            torch.cuda.empty_cache()

    end = time.time()
    print(f"Training time: {end-start}")
    return net
