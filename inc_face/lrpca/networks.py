import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

datatype = torch.float32


class LRPCANet(nn.Module):
    def __init__(self, r, maxIt, ths_initial, step_initial, device):
        super(type(self), self).__init__()
        self.ths_v = [
            nn.Parameter(
                Variable(
                    torch.tensor(ths_initial, dtype=datatype, device=device),
                    requires_grad=True,
                )
            )
            for t in range(maxIt)
        ]
        self.step = [
            nn.Parameter(
                Variable(
                    torch.tensor(step_initial, dtype=datatype, device=device),
                    requires_grad=True,
                )
            )
            for t in range(maxIt)
        ]
        self.ths_backup = [
            torch.tensor(0.0, dtype=datatype, device=device) for t in range(maxIt)
        ]
        self.device = device
        self.maxIt = maxIt
        self.max_iter = maxIt

        self.r = r

    def thre(self, inputs, threshold):
        out = torch.sign(inputs) * torch.max(
            torch.abs(inputs) - threshold,
            torch.zeros([1, 1], dtype=datatype, device=self.device),
        )
        return out

    def forward(self, X_orig, Y0_t, num_l):
        ## Initialization
        S_t = self.thre(Y0_t, self.ths_v[0])
        L, Sigma, R = torch.svd_lowrank(Y0_t - S_t, q=self.r, niter=4)
        Sigsqrt = torch.diag(torch.sqrt(Sigma))
        U_t = torch.mm(L, Sigsqrt)
        V_t = torch.mm(R, Sigsqrt)

        ## Main Loop in LRPCA
        for t in range(1, num_l):
            YmUV = Y0_t - torch.mm(U_t, V_t.t())
            S_t = self.thre(YmUV, self.ths_v[t])
            E_t = YmUV - S_t
            Vkernel = torch.inverse(V_t.t() @ V_t)
            Ukernel = torch.inverse(U_t.t() @ U_t)
            Unew = U_t + self.step[t] * (torch.mm(E_t, V_t) @ Vkernel)
            Vnew = V_t + self.step[t] * (torch.mm(U_t.t(), E_t).t() @ Ukernel)
            U_t = Unew
            V_t = Vnew

        ## loss function in training
        loss = (torch.mm(U_t, V_t.t()) - X_orig).norm() / torch.norm(X_orig)
        return loss

    def InitializeThs(self, en_l):
        self.ths_v[en_l].data = torch.clone(self.ths_v[en_l - 1].data * 0.9)
        self.step[en_l].data = torch.clone(self.step[en_l - 1].data * 0.5)

    def CheckNegative(self):
        isNegative = False
        for t in range(self.maxIt):
            if self.ths_v[t].data < 0:
                isNegative = True
        if isNegative:
            for t in range(self.maxIt):
                self.ths_v[t].data = torch.clone(self.ths_backup[t])
        else:
            for t in range(self.maxIt):
                self.ths_backup[t] = torch.clone(self.ths_v[t].data)
        return isNegative

    def EnableSingleLayer(self, en_l):
        for t in range(self.maxIt):
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False
        self.ths_v[en_l].requires_grad = True
        self.step[en_l].requires_grad = True

    def EnableLayers(self, num_l):
        for t in range(num_l):
            self.ths_v[t].requires_grad = True
            self.step[t].requires_grad = True
        for t in range(num_l, self.maxIt):
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False
