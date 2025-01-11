import torch
import math


class SyntheticData:
    """
    A class for generating synthetic data for robust matrix completion problems.
    It can generate both fully observed and partially observed matrices based on the
    specified parameters.
    Attributes:
        d1 (int): The number of rows in the generated matrix.
        d2 (int): The number of columns in the generated matrix.
        r (int): The rank used to generate the low-rank matrices.
        alpha (float): The fraction of entries in the matrix to be corrupted with noise.
        p (float): The fraction of entries to be observed when generating partially
            observed matrices. If p < 1, a partially observed matrix is generated;
            otherwise, a fully observed matrix is generated.
        device (torch.device): The device where all tensors will be created and
            calculations will be performed. Default is the CPU.
    Methods:
        new():
            Determines whether to generate a partially observed or a fully observed
            matrix, depending on the value of the p attribute. Returns the outputs
            from the corresponding generation method.

    """

    def __init__(self, d1, d2, r, alpha, p, device) -> None:
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.alpha = alpha
        self.p = p
        self.device = device

    def new(self):
        if self.p < 1:
            return self.generate_partial()
        else:
            return self.generate_full()

    def generate_partial(self):
        U = torch.randn(self.d1, self.r, device=self.device) / math.sqrt(self.d1)
        V = torch.randn(self.d2, self.r, device=self.device) / math.sqrt(self.d2)
        X_orig = torch.mm(U, V.t())

        # generate observed matrix
        idx = torch.randperm(self.d1 * self.d2, device=self.device)
        idx = idx[: math.floor(self.alpha * self.d1 * self.d2)]
        Y = X_orig.clone()
        Y = Y.reshape(-1)

        # generate noise
        a = torch.mean(torch.abs(Y))
        S = torch.rand(len(idx), device=self.device)
        S = 2 * a * (S - 0.5)
        Y[idx] = Y[idx] + S
        Y = Y.reshape((self.d1, self.d2))
        omega_id = torch.randperm(self.d1 * self.d2, device=self.device)[
            : int(self.d1 * self.d2 * (self.p))
        ]
        omega = torch.zeros(self.d1 * self.d2, device=self.device)
        omega.view(-1)[omega_id] = 1
        Y = omega.view(self.d1, self.d2) * Y

        return U, V, Y, omega.view(self.d1, self.d2)

    def generate_full(self):
        U = torch.randn(self.d1, self.r, device=self.device) / math.sqrt(self.d1)
        V = torch.randn(self.d2, self.r, device=self.device) / math.sqrt(self.d2)

        idx = torch.randperm(self.d1 * self.d2, device=self.device)
        idx = idx[: math.floor(self.alpha * self.d1 * self.d2)]
        Y = torch.mm(U, V.t())
        Y = Y.reshape(-1)
        a = torch.mean(torch.abs(Y))
        S = torch.rand(len(idx), device=self.device)
        S = 2 * a * (S - 0.5)
        Y[idx] = Y[idx] + S
        Y = Y.reshape((self.d1, self.d2))
        return U, V, Y, None


if __name__ == "__main__":
    data = SyntheticData(
        3000,
        3000,
        5,
        0.1,
        0.1,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    U, V, Y, omega = data.new()
