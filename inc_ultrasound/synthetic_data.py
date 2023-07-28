import torch
import math

class DataGenerator:
    def __init__(self, d1, d2, r, alpha, p, device) -> None:
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.alpha = alpha
        self.p = p
        self.device = device


    def new(self):
        U = torch.randn(self.d1, self.r, device=self.device)/math.sqrt(self.d1)
        V = torch.randn(self.d2, self.r, device=self.device)/math.sqrt(self.d2)
        X_orig = torch.mm(U, V.t())

        # generate observed matrix
        idx = torch.randperm(self.d1*self.d2, device=self.device)
        idx = idx[:math.floor(self.alpha * self.d1*self.d2)]
        Y = X_orig.clone()
        Y = Y.reshape(-1)

        # generate noise
        a = torch.mean(torch.abs(Y))
        S = torch.rand(len(idx), device=self.device)
        S = 2*a* (S - 0.5)
        Y[idx] = Y[idx] + S
        Y = Y.reshape((self.d1, self.d2))
        omega_id = torch.randperm(self.d1*self.d2, device=self.device)[:int(self.d1*self.d2*(self.p))]
        omega = torch.zeros(self.d1*self.d2, device=self.device)
        omega.view(-1)[omega_id] = 1
        Y = omega.view(self.d1, self.d2) * Y

        return U,V, Y, omega.view(self.d1, self.d2)
    
def get_sup_val(num_iter, n, r, alpha, p, device):

    max_value = torch.tensor(0,device=device)
    data = DataGenerator(n,n,r,alpha,p, device)
    for i in range(num_iter):
        U,V,_,_ = data.new()
        X_orig = torch.mm(U, V.t())
        X_max = torch.max(X_orig)
        if X_max > max_value:
            max_value = X_max
        if i%1000 == 0:
            print(f'iteration {i} - max_value: {max_value}')
    return max_value

if __name__ == "__main__":
    get_sup_val(5000, 5000, 5, 0.1, 0.2, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        
    
