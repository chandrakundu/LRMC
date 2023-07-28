import torch
import torch.nn as nn
from torch.autograd import Variable
datatype = torch.float32


class LRMCNet(nn.Module):
    def __init__(self, r, p, max_iter, zeta0, eta0, device):
        super(type(self),self).__init__()        
        self.zeta 		= [nn.Parameter(Variable(torch.tensor(zeta0, dtype=datatype, device = device), requires_grad=True)) for t in range(max_iter)]
        self.eta 		= [nn.Parameter(Variable(torch.tensor(eta0, dtype=datatype, device = device), requires_grad=True)) for t in range(max_iter)]
        self.zeta_backup	= [torch.tensor(0.0, dtype=datatype, device = device) for t in range(max_iter)]
        # self.eta_backup	    = [torch.tensor(0.0, dtype=datatype, device = device) for t in range(max_iter)]

        self.r 			= r
        self.p 			= p
        self.max_iter 	= max_iter        
        self.device 	= device
        self.loss_fn    = nn.MSELoss()

    # def soft_thres(self, inputs, threshold):
    #     out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], device=self.device) )
    #     return out
    
    def soft_thres(self, x, lambd):
        return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)

    def forward(self, Lfista, Sfista, Y, omega, num_l):
        S_new = Y - self.soft_thres(Y, self.zeta[0])*self.eta[0]
        U0, Sigma0, V0 = torch.svd_lowrank(S_new,self.r,niter=4)
        Sigma_Sqrt = torch.diag(torch.sqrt(Sigma0))
        L = torch.mm(U0,Sigma_Sqrt)
        R = torch.mm(V0, Sigma_Sqrt)
        
        for t in range(1, num_l):
            X_new = torch.mm(L,R.t())
            diff = Y - omega*X_new
            S_new = self.soft_thres(diff, self.zeta[t])            
            midterm = (S_new - diff)
            RRinv = torch.inverse(torch.mm(R.t(),R))    
            LLinv = torch.inverse(torch.mm(L.t(),L))
            Lk = L - self.eta[t]*torch.mm(torch.mm(midterm, R), RRinv)
            Rk = R - self.eta[t]*torch.mm(torch.mm(midterm.t(), L), LLinv)
            L = Lk
            R = Rk
            
        loss = 0.5*self.loss_fn(torch.mm(L,R.t()), Lfista) + 0.5*self.loss_fn(S_new, Sfista)
        return loss
        
        
    def initalize_zeta(self, en_l):
        self.zeta[en_l].data = torch.clone(self.zeta[en_l-1].data * 0.9)

    def initalize_eta(self, en_l):
        self.eta[en_l].data = torch.clone(self.eta[en_l-1].data)
        # self.eta[en_l].data = torch.tensor(1.0/self.p, dtype=datatype, device = self.device)
        
        # if en_l > 8:
        #     self.eta[en_l].data = torch.tensor(1.0/self.p, dtype=datatype, device = self.device)

        # # this elif is for the case of recoveribility when alpha is big
        # elif en_l > 20:
        #     eta_add = 1 + torch.sqrt(1/(200-en_l))
        #     self.eta[en_l].data = en_l + eta_add
        # else:
        #     eta_add = torch.clone((1/self.p-1)*(self.eta[0]/7))
        #     self.eta[en_l].data = torch.clone(self.eta[en_l-1].data + eta_add)
        
        
        

    def check_negative(self):
        isNegative = False;
        for t in range(self.max_iter):
            if(self.zeta[t].data < 0):
                isNegative = True;
        if(isNegative):
            for t in range(self.max_iter):
                self.zeta[t].data = torch.clone(self.zeta_backup[t])
        else:
            for t in range(self.max_iter):
                self.zeta_backup[t] = torch.clone(self.zeta[t].data)
        return isNegative;

    def enable_single_layer(self,en_l):
        for t in range(self.max_iter): 
            self.zeta[t].requires_grad = False
            self.eta[t].requires_grad = False
        self.zeta[en_l].requires_grad = True
        self.eta[en_l].requires_grad = True

    def enable_layers(self, num_l):
        for t in range(num_l): 
            self.zeta[t].requires_grad = True
            self.eta[t].requires_grad = True
        for t in range(num_l,self.max_iter): 
            self.zeta[t].requires_grad = False
            self.eta[t].requires_grad = False



class LRPCANet(nn.Module):
    def __init__(self, r, maxIt, ths_initial, step_initial, device):
        super(type(self),self).__init__()
        self.ths_v 		= [nn.Parameter(Variable(torch.tensor(ths_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
        self.step 		= [nn.Parameter(Variable(torch.tensor(step_initial, dtype=datatype, device = device), requires_grad=True)) for t in range(maxIt)]
        self.ths_backup	= [torch.tensor(0.0, dtype=datatype, device = device) for t in range(maxIt)]
        self.device = device
        self.maxIt = maxIt
        self.max_iter = maxIt
        self.r = r
        self.loss_fn    = nn.MSELoss()
        

    def thre(self, inputs, threshold):
        out = torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros([1, 1], dtype=datatype, device=self.device) )
        return out

    def forward(self, Y0_t, Lfista, Sfista, num_l):
        
        ## Initialization
        S_t =  self.thre(Y0_t, self.ths_v[0])
        L, Sigma, R = torch.svd_lowrank(Y0_t-S_t, q = self.r, niter = 4)
        Sigsqrt = torch.diag(torch.sqrt(Sigma))
        U_t = torch.mm(L, Sigsqrt)
        V_t = torch.mm(R, Sigsqrt) 

        ## Main Loop in LRPCA
        for t in range(1, num_l):
            YmUV = Y0_t - torch.mm(U_t, V_t.t())
            S_t =  self.thre(YmUV, self.ths_v[t])
            E_t = YmUV - S_t 
            Vkernel = torch.inverse(V_t.t() @ V_t)
            Ukernel = torch.inverse(U_t.t() @ U_t)
            Unew = U_t + self.step[t] * (torch.mm(E_t,V_t) @ Vkernel)
            Vnew = V_t + self.step[t] * (torch.mm(U_t.t(),E_t).t() @ Ukernel)
            U_t = Unew
            V_t = Vnew
                        

        ## loss function in training
        loss = 0.5*self.loss_fn(torch.mm(U_t,V_t.t()), Lfista) + 0.5*self.loss_fn(S_t, Sfista)	

        return loss

    def InitializeThs(self, en_l):
        self.ths_v[en_l].data = torch.clone(self.ths_v[en_l-1].data * 0.9)
        
    def CheckNegative(self):
        isNegative = False;
        for t in range(self.maxIt):
            if(self.ths_v[t].data < 0):
                isNegative = True;
        if(isNegative):
            for t in range(self.maxIt):
                self.ths_v[t].data = torch.clone(self.ths_backup[t])
        else:
            for t in range(self.maxIt):
                self.ths_backup[t] = torch.clone(self.ths_v[t].data)
        return isNegative;

    def EnableSingleLayer(self,en_l):
        for t in range(self.maxIt): 
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False
        self.ths_v[en_l].requires_grad = True
        self.step[en_l].requires_grad = True

    def EnableLayers(self, num_l):
        for t in range(num_l): 
            self.ths_v[t].requires_grad = True
            self.step[t].requires_grad = True
        for t in range(num_l,self.maxIt): 
            self.ths_v[t].requires_grad = False
            self.step[t].requires_grad = False
