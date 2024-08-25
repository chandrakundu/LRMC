import torch
import torch.nn as nn
from torch.autograd import Variable

datatype = torch.float32


class LRMCNet(nn.Module):
    def __init__(self, d1, d2, r, alpha, p, max_iter, zeta0, eta0, device):
        super(type(self), self).__init__()
        self.zeta = [
            nn.Parameter(
                Variable(
                    torch.tensor(zeta0[t], dtype=datatype, device=device),
                    requires_grad=True,
                )
            )
            for t in range(max_iter)
        ]
        self.eta = [
            nn.Parameter(
                Variable(
                    torch.tensor(eta0[t], dtype=datatype, device=device),
                    requires_grad=True,
                )
            )
            for t in range(max_iter)
        ]
        self.zeta_backup = [
            torch.tensor(0.0, dtype=datatype, device=device) for t in range(max_iter)
        ]
        # self.eta_backup	    = [torch.tensor(0.0, dtype=datatype, device = device) for t in range(max_iter)]

        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.alpha = alpha
        self.p = p
        self.max_iter = max_iter
        self.device = device

    # def soft_thres(self, inputs, threshold):
    #     out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], device=self.device) )
    #     return out

    def soft_thres(self, x, lambd):
        return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)

    def forward(self, U, V, Y, omega, num_l):
        S0 = (Y - self.soft_thres(Y, self.zeta[0])) * self.eta[0]
        U0, Sigma0, V0 = torch.svd_lowrank(S0, self.r, niter=4)
        Sigma_Sqrt = torch.diag(torch.sqrt(Sigma0))
        L = torch.mm(U0, Sigma_Sqrt)
        R = torch.mm(V0, Sigma_Sqrt)

        for t in range(1, num_l):
            X_new = torch.mm(L, R.t())
            diff = Y - omega * X_new
            S_new = self.soft_thres(diff, self.zeta[t])
            midterm = S_new - diff
            RRinv = torch.inverse(torch.mm(R.t(), R))
            LLinv = torch.inverse(torch.mm(L.t(), L))
            Lk = L - self.eta[t] * torch.mm(torch.mm(midterm, R), RRinv)
            Rk = R - self.eta[t] * torch.mm(torch.mm(midterm.t(), L), LLinv)
            L = Lk
            R = Rk

        X_orig = torch.mm(U, V.t())
        loss = torch.norm(X_orig - torch.mm(L, R.t())) / torch.norm(X_orig)
        return loss

    def initalize_zeta(self, en_l):
        self.zeta[en_l].data = torch.clone(self.zeta[en_l - 1].data * 0.6)

    def initalize_eta(self, en_l):
        # self.eta[en_l].data = torch.clone(self.eta[en_l-1].data)
        # self.eta[en_l].data = torch.tensor(1.0/self.p, dtype=datatype, device = self.device)

        if en_l > 0 and en_l <= 1:
            # set equla 1.3
            self.eta[en_l].data = torch.tensor(1.3, dtype=datatype, device=self.device)
        elif en_l > 1 and en_l <= 10:
            self.eta[en_l].data = torch.clone(self.eta[en_l - 1].data * 1.3)
        elif en_l > 10:
            self.eta[en_l].data = torch.tensor(
                1.0 / self.p, dtype=datatype, device=self.device
            )

        # this elif is for the case of recoveribility when alpha is big
        elif en_l > 20:
            eta_add = 1 + torch.sqrt(1 / (200 - en_l))
            self.eta[en_l].data = en_l + eta_add
        else:
            eta_add = torch.clone((1 / self.p - 1) * (self.eta[0] / 7))
            self.eta[en_l].data = torch.clone(self.eta[en_l - 1].data + eta_add)

    def check_negative(self):
        isNegative = False
        for t in range(self.max_iter):
            if self.zeta[t].data < 0:
                isNegative = True
        if isNegative:
            for t in range(self.max_iter):
                self.zeta[t].data = torch.clone(self.zeta_backup[t])
        else:
            for t in range(self.max_iter):
                self.zeta_backup[t] = torch.clone(self.zeta[t].data)
        return isNegative

    def enable_single_layer(self, en_l):
        for t in range(self.max_iter):
            self.zeta[t].requires_grad = False
            self.eta[t].requires_grad = False
        self.zeta[en_l].requires_grad = True
        self.eta[en_l].requires_grad = True

    def enable_layers(self, num_l):
        for t in range(num_l):
            self.zeta[t].requires_grad = True
            self.eta[t].requires_grad = True
        for t in range(num_l, self.max_iter):
            self.zeta[t].requires_grad = False
            self.eta[t].requires_grad = False
