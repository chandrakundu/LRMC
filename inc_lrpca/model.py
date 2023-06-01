import torch
import time 
import os
import scipy.io as sio
import numpy as np
import gc

def train(net, data, nepoch_pre, nepoch_full, lr_init, lr_frac_zeta=5000.0, lr_frac_eta=10.0):
    

    optimizers = []
    # default lr frac for synthetic data is 10000.0
    # default lr frac is same for zeta and eta
    
    for t in range(net.max_iter):
        optimizer = torch.optim.SGD({net.ths_v[t]}, lr=lr_init/lr_frac_zeta)
        optimizer.add_param_group({'params': net.step[t], 'lr': lr_init / lr_frac_eta})  
        optimizers.append(optimizer)

    start = time.time()
    for stage in range(net.max_iter):        
        
        print(f'Layer {stage} Pre-training ========================')
        
        if stage>6:
            nepoch_full = nepoch_pre

        if (stage>0):
            optimizers[stage].param_groups[0]['lr'] = net.ths_v[stage-1].data * lr_init /lr_frac_zeta

        # if stage>5:
        #     nepoch_full = 501
        #     nepoch_pre = 101
        #     optimizers[stage].param_groups[1]['lr'] = 0.001

        # if stage<7:
        #     nepoch_full = 501
        #     nepoch_pre = 301
            
        total_loss = 0
        for epoch in range(nepoch_pre):
            U, V, Y = data.new()

            for i in range(net.max_iter):
                optimizers[i].zero_grad()

            net.EnableSingleLayer(stage)
            if stage > 0:
                net.InitializeThs(stage)

            loss = net(Y, U, V, stage+1)
            total_loss += loss.item()
            loss.backward()
            optimizers[stage].step()

            if epoch % 10 == 0:
                if net.CheckNegative():
                    print("Negative zeta detected and reset to previous value")

            if epoch % 100 == 0 and epoch > 0:
                print(f'Pretrain: Layer {stage} Epoch {epoch}: Loss {loss.item()} average loss: {total_loss/(epoch+1)}')

            # if epoch % 10 == 0:
            #     print(f'Pretrain: Layer {stage} Epoch {epoch}: Loss {loss.item()} average loss: {total_loss/(epoch+1)}')
                

            del U, V, Y, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        # print('----- parameter update -----')
        # print(f'zeta: {net.zeta[stage].data} and eta: {net.eta[stage].data}')
        # print(f'eta = {[net.eta[j].data.item() for j in range(stage+1)]}')
        # print(f'zeta = {[net.zeta[j].data.item() for j in range(stage+1)]}')

        if stage == 0:
            continue

        print(f'Layer {stage} Full Training ========================')
        total_loss = 0
        for epoch in range(nepoch_full):
            U, V, Y = data.new()

            for i in range(net.max_iter):
                optimizers[i].zero_grad()

            net.EnableLayers(stage+1)
            loss = net(Y, U, V, stage+1)
            total_loss += loss.item()
            loss.backward()

            for i in range(stage+1):
                optimizers[i].step()

            if epoch % 100 == 0 and epoch > 0:
                print(f'Fulltrain: Layer {stage} Epoch {epoch}: Loss {loss.item()} average loss: {total_loss/(epoch+1)}')

            del U, V, Y, loss
            gc.collect()
            torch.cuda.empty_cache()

              
        # print etas and zetas in one line
        # print('----- parameter update -----')
        # print(f'zeta: {net.zeta[stage].data} and eta: {net.eta[stage].data}')
        # print(f'etas: {[net.eta[j].data.item() for j in range(stage+1)]}')
        # print(f'zetas: {[net.zeta[j].data.item() for j in range(stage+1)]}')
            
    end = time.time()
    print(f'Training time: {end-start}')
    return net


def save(net, fname_ext="", dirname='saved_models_lrpca'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f'Created directory {dirname}. Now saving model...')
    result_zeta = np.zeros((net.max_iter,))
    result_eta = np.zeros((net.max_iter,))
    for i in range(net.max_iter):
        result_zeta[i] 	= net.zeta[i].data.cpu().numpy()
        result_eta[i] 	= net.eta[i].data.cpu().numpy()

    spath = f'{dirname}/lrpca_n{net.d1}_r{net.r}_alpha{net.alpha}_{fname_ext}.mat'
    sio.savemat(spath, {'zeta':result_zeta, 'eta':result_eta})  
    print(f'Model saved to {spath}') 
            



            

    