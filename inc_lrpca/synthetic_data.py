import torch
import math

datatype = torch.float32

class DataGenerator:
    def __init__(self, d1, d2, r, alpha, device) -> None:
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.alpha = alpha
        self.device = device


    def new(self):
        U0_t 		= torch.randn(self.d1,self.r,dtype = datatype, device = self.device)/math.sqrt(self.d1)
        V0_t 		= torch.randn(self.d1,self.r,dtype = datatype, device = self.device)/math.sqrt(self.d2)
        idx 		= torch.randperm(self.d1*self.d2, device = self.device)
        idx 		= idx[:math.floor(self.alpha * self.d1*self.d2)]
        Y0_t 		= torch.mm(U0_t,V0_t.t())
        Y0_t 		= Y0_t.reshape(-1)
        s_range		= torch.mean(torch.abs(Y0_t))
        S0_t 		= torch.rand(len(idx), dtype = datatype, device = self.device)
        S0_t 		= s_range * (2.0 * S0_t - 1.0)
        Y0_t[idx] 	= Y0_t[idx] + S0_t
        Y0_t 		= Y0_t.reshape((self.d1,self.d2))
        return U0_t, V0_t, Y0_t
    
def get_sup_val(num_iter, n, r, alpha, device):

    max_value = torch.tensor(0,device=device)
    data = DataGenerator(n,n,r,alpha, device)
    for i in range(num_iter):
        X_orig,_,_ = data.new()
        X_max = torch.max(X_orig)
        if X_max > max_value:
            max_value = X_max
        if i%1000 == 0:
            print(f'iteration {i} - max_value: {max_value}')
    return max_value

if __name__ == "__main__":
    get_sup_val(5000, 5000, 5, 0.1, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        
    
