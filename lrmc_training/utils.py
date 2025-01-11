import numpy as np
import scipy.io as sio
import os


def save(net, filepath):
    """
    Save trained parameters (zeta and eta) into .mat format.
    Args:
        net: Trained neural network model containing zeta and eta parameters
        filepath (str): Path where to save the .mat file
    Notes:
        - Parameters saved:
            - zeta: Array of zeta values for each iteration
            - eta: Array of eta values for each iteration
    """

    result_zeta = np.zeros((net.max_iter,))
    result_eta = np.zeros((net.max_iter,))
    for i in range(net.max_iter):
        result_zeta[i] = net.zeta[i].data.cpu().numpy()
        result_eta[i] = net.eta[i].data.cpu().numpy()

    folder = os.path.dirname(filepath)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)

    sio.savemat(filepath, {"zeta": result_zeta, "eta": result_eta})
    print(f"Model saved to {filepath}")
