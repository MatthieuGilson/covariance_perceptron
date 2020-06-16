#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:03:18 2019

@author: gilsonmatthieu
"""

import numpy as np
import matplotlib.pyplot as pp



#%%
# init

work_dir = 'mnist5digits_RNN/'

grph_fmt = 'eps'


cols_gr = []
for ii in [2,1,0]:
    cols_gr += [[ii*0.3,ii*0.3,ii*0.3]]


#%% data
data_dir = 'mnist_data/'
train_data = np.load(data_dir+'train_data.npy')
test_data = np.load(data_dir+'test_data.npy')
train_label = np.load(data_dir+'train_labels.npy')
test_label = np.load(data_dir+'test_labels.npy')

nrows = 9 # size of images
n_digits = 5 # number of distinct digits

# create labels for each motion direction
train_label = np.concatenate((train_label,train_label+n_digits))
test_label = np.concatenate((test_label,test_label+n_digits))

n_cat = np.unique(train_label).size # number of categories

# double the number of input samples (for two motion directions)
n_train = train_label.size
n_test = test_label.size

#v_n_pat = np.array([50000],dtype=np.int)
v_n_pat = np.array([50000],dtype=np.int)
n_n_pat = v_n_pat.size


#%%
M = nrows * 2

n_shift = 1
T = nrows + n_shift # move all digit till end

# layer of unfolded RNN, batch calculations
class layer_batch:            
    def f(self, x):
        # sigmoid
#        return 1 / (1 + np.exp(-x))
        # tanh
        ex1 = np.exp(x)
        ex2 = np.exp(-x)
        return (ex1 - ex2)/ (ex1 + ex2)
    
    def df(self, x):
        # sigmoid
#        return self.f(x) * (1 - self.f(x))
        # tanh
        return 1 - self.f(x)**2

    def fwd(self, x, y, A, B, C):
        y1 = np.ones(y.shape)
        self.arg_y = np.einsum('ij, kj -> ki', A, y[:,:-1]) + np.einsum('ij, kj -> ki', B, x)
        y1[:,:-1] = self.f(self.arg_y)
        self.arg_z = np.einsum('ij, kj -> ki', C, y1)
        z1 = self.f(self.arg_z)
        return y1, z1
    
    # returns weight updates for B and A
    def bkwd(self, x, y, A, B, C, err, bool_first_layer):
        y1, z1 = self.fwd(x, y, A, B, C)
        n_samples = x.shape[0]
        M = x.shape[1]
        N = y.shape[1]
        O = z1.shape[1]
        if bool_first_layer: # k is pattern index
            e1 = np.einsum('ij, ki -> kj', C, err)[:,:-1] # equiv to C.T
            dA = np.einsum('kil, kjl -> kij', (e1 * self.df(self.arg_y)).reshape([n_samples,N-1,1]), y[:,:-1].reshape([n_samples,N-1,1]))
            dB = np.einsum('kil, kjl -> kij', (e1 * self.df(self.arg_y)).reshape([n_samples,N-1,1]), x.reshape([n_samples,M,1]))
            dC = np.einsum('kil, kjl -> kij', (err * self.df(self.arg_z)).reshape([n_samples,O,1]), y1.reshape([n_samples,N,1]))
            return dA, dB, dC, np.einsum('ij, ki -> kj', A, e1)
        else:
            dA = np.einsum('kil, kjl -> kij', (err * self.df(self.arg_y)).reshape([n_samples,N-1,1]), y[:,:-1].reshape([n_samples,N-1,1]))
            dB = np.einsum('kil, kjl -> kij', (err * self.df(self.arg_y)).reshape([n_samples,N-1,1]), x.reshape([n_samples,M,1]))
            return dA, dB, np.zeros(C.shape), np.einsum('ij, ki -> kj', A, err)


# recurrent neural network (batch)       
class rnn():
    def __init__(self, M, N, O, T, L, A=[], B=[], C=[]):
        self.M = M # number of inputs
        self.N = N # number of reservoir units
        self.O = O # number of outputs
        self.T = T # number of layers in time
        self.L = L # number of layers to backpropagate error
        # initial connectivities
        if len(A)==0:
            self.A = np.random.randn(N,N) * 0.01 # input weight matrix
        else:
            assert(A.shape==(N,N))
            self.A = A
        if len(B)==0:
            self.B = np.random.randn(N,M+1) * 0.1 # recurrent weight matrix
        else:
            assert(B.shape==(N,M+1))
            self.B = B
        if len(C)==0:
            self.C = np.random.randn(O,N+1) * 0.1 # input weight matrix
        else:
            assert(C.shape==(O,N+1))
            self.C = C        
        # learning rates
        self.set_eta(0.001)
        
    def set_eta(self, eta):
        self.etaA = eta
        self.etaB = eta
        self.etaC = eta
    
    def pred(self, xt):
        # input is xt + bias = 1
        assert(xt.shape[1:3]==(self.M,self.T))
        n_samples = xt.shape[0]
        self.xt = np.zeros([n_samples,self.M+1,self.T])
        self.xt[:,:self.M,:] = xt
        self.xt[:,self.M,:] = 1
        # reservoir yt + bias = 1
        self.yt = np.zeros([n_samples,self.N+1,self.T])
        self.yt[:,self.N,:] = 1
        # output zt
        self.zt = np.zeros([n_samples,self.O,self.T])
        # calculate forward propagation
        layer = layer_batch()
        for t in range(self.T):
            self.yt[:,:,t], self.zt[:,:,t] = layer.fwd(self.xt[:,:,t], self.yt[:,:,t-1], self.A, self.B, self.C)
        return self.zt

    def bptt(self, xt, objt):
        # calculate forward propagation
        self.pred(xt)
        # calculate error for each time step
        errt = objt - self.zt
        # calculate weight update for each time steps, going for each L steps backwards
        dA = np.zeros(self.A.shape)
        dB = np.zeros(self.B.shape)
        dC = np.zeros(self.C.shape)
        layer = layer_batch()
        for t in range(self.T):
            if not np.any(np.isnan(errt[0,:,t])):
                err_tmp = errt[:,:,t] # error on output at time t
                for l in range(t,max(t-self.L,-1),-1):
                    dA_tmp, dB_tmp, dC_tmp, err_tmp = layer.bkwd(self.xt[:,:,t], self.yt[:,:,t-1], self.A, self.B, self.C, err_tmp, l==t)
                    # sum over samples
                    dA += dA_tmp.sum(0)
                    dB += dB_tmp.sum(0)
                    dC += dC_tmp.sum(0)
        # update the weights with the sum of all updates
        self.A += self.etaA * dA
        self.B += self.etaB * dB
        self.C += self.etaC * dC

    def err(self, xt, objt):
        self.pred(xt)
        validt = np.logical_not(np.isnan(objt[0,0,:]))
        return np.linalg.sum((self.zt[:,:,validt] - objt[:,:,validt])**2, axis=1)

    def score(self, xt, labels, T0):
        self.pred(xt)
        n_samples = xt.shape[0]
        # evaluation based on output with largest value (discarding time points before T0)
        z = self.zt[:,:,T0:].mean(2)
        pred_labels = np.argmax(z, axis=1)
        # accuracy
        acc_tmp = np.sum(labels==pred_labels)
        # confusion matrix
        CM_tmp = np.zeros([n_cat,n_cat]) # confusion matrix
        for i_pat in np.arange(n_samples):
            CM_tmp[labels[i_pat],pred_labels[i_pat]] += 1
        # returns accuracy and confusion matrix
        return acc_tmp / n_samples, CM_tmp
        
# create time series of input receptor neurons from MNIST images
def get_input(data_arg, i_pat_arg, n_arg):
    # create left/right moving
    I_tmp = np.zeros([M,T])
    if i_pat_arg<n_arg/2:
        # left row: early
        I_tmp[:int(M/2),:-n_shift] = data_arg[i_pat_arg,:,:]
        # right row: late
        I_tmp[int(M/2):,n_shift:] = data_arg[i_pat_arg,:,:]
    else:
        # left row: late
        I_tmp[:int(M/2),n_shift:] = data_arg[i_pat_arg-int(n_arg/2),:,:]
        # right row: early
        I_tmp[int(M/2):,:-n_shift] = data_arg[i_pat_arg-int(n_arg/2),:,:]
    return I_tmp

# create batch of data + labels from subset of indices
def create_batch(data_arg, n_arg, label_arg, T_arg, ind_arg):
    # features
    data_batch = np.zeros([ind_arg.size,M,T])
    # objective
    obj_batch = np.zeros([ind_arg.size,O,T])
    # collect all desired patterns
    for ii, i_pat in enumerate(ind_arg):
        data_batch[ii,:,:] = get_input(data_arg, i_pat, n_arg)
        obj_batch[ii,label_arg[i_pat],:] = 1
    # discard objective for initial time points
    obj_batch[:,:,:T_arg] = np.nan
    return data_batch, obj_batch


#%%
# classification
O = n_cat
N = 6  # hidden neurons with matched number of resources
#N = 10 # hidden neurons

L = 5
T0 = max(0,L-1) # time steps to ignore error

n_opt = 20
batch_size = 1000

i_n_pat = 0
n_rep = 10

    
# mean, cov, cov with m cov entries (randomly chosen) instead of m(m-1)/2
perf = np.zeros([n_rep,2])
CM = np.zeros([n_rep,2,n_cat,n_cat])


for i_rep in range(n_rep):
    print('rep', i_rep)
    
    # select subset of samples
    n_pat_train = v_n_pat[i_n_pat]
    n_pat_test = int(n_pat_train/10)
    
    # construct subset of training patterns
    subset_train = np.zeros([n_train], dtype=np.bool)
    subset_train[np.random.rand(n_train)<n_pat_train/n_train] = True
    while not subset_train.sum()==n_pat_train:
        if subset_train.sum()<n_pat_train:
            ind_false = np.argwhere(np.logical_not(subset_train))
            subset_train[ind_false[np.random.randint(n_train-subset_train.sum())]] = True
        else:
            ind_true = np.argwhere(subset_train)
            subset_train[ind_true[np.random.randint(subset_train.sum())]] = False
    
    subset_test = np.zeros([n_test], dtype=np.bool)
    subset_test[np.random.rand(n_test)<n_pat_test/n_test] = True
    while not subset_test.sum()==n_pat_test:
        if subset_test.sum()<n_pat_test:
            ind_false = np.argwhere(np.logical_not(subset_test))
            subset_test[ind_false[np.random.randint(n_test-subset_test.sum())]] = True
        else:
            ind_true = np.argwhere(subset_test)
            subset_test[ind_true[np.random.randint(subset_test.sum())]] = False
    
    # optimization loop
    clf_opt = rnn(M,N,O,T,L)
    eta = 1 / n_pat_train
    clf_opt.set_eta(eta)
    
    perf_tmp = np.zeros([n_opt+1,2])
    CM_tmp = np.zeros([n_opt+1,2,n_cat,n_cat])

    data_batch, tmp = create_batch(train_data, n_train, train_label, T0, np.arange(n_train)[subset_train])
    perf_tmp[0,0], CM_tmp[0,0,:,:] = clf_opt.score(data_batch, train_label[subset_train], T0)
    
    data_batch, tmp = create_batch(test_data, n_test, test_label, T0, np.arange(n_test)[subset_test])
    perf_tmp[0,1], CM_tmp[0,1,:,:] = clf_opt.score(data_batch, test_label[subset_test], T0)
    print('before:', perf_tmp[0,:])

    for i_opt in range(n_opt):

        ind_train = np.random.permutation(np.arange(n_train)[subset_train])
        # loop over batches of indices
        for i_batch in range(int(ind_train.size/batch_size)):
            
            ind_batch = ind_train[i_batch*batch_size:(i_batch+1)*batch_size]
            data_batch, obj_batch = create_batch(train_data, n_train, train_label, T0, ind_batch)
                
            # BPTT
            clf_opt.bptt(data_batch, obj_batch)

        # test perf
        data_batch, tmp = create_batch(train_data, n_train, train_label, T0, np.arange(n_train)[subset_train])
        perf_tmp[i_opt+1,0], CM_tmp[i_opt+1,0,:,:] = clf_opt.score(data_batch, train_label[subset_train], T0)
        
        data_batch, tmp = create_batch(test_data, n_test, test_label, T0, np.arange(n_test)[subset_test])
        perf_tmp[i_opt+1,1], CM_tmp[i_opt+1,1,:,:] = clf_opt.score(data_batch, test_label[subset_test], T0)

        print('opt', i_opt, ':', perf_tmp[i_opt+1,:])
        
        if i_opt>n_opt/2:
            eta *= 0.95
            clf_opt.set_eta(eta)

    # take best training score
    i_opt_best = np.argmax(perf_tmp[:,0])
    perf[i_rep,:] = perf_tmp[i_opt_best,:]
    CM[i_rep,:,:,:] = CM_tmp[i_opt_best,:,:,:]
    
    print(perf[i_rep,:])


np.save(work_dir+'perf.npy',perf)
np.save(work_dir+'CM.npy',CM)


#%%    
# plots
    
pp.figure()
pp.violinplot(perf, positions=range(2))
pp.plot([-0.5,1.5],[1/n_cat,1/n_cat],'--k')
pp.axis(ymin=0,ymax=1)
pp.xticks([0,1],['train','test'])
pp.ylabel('perf')
pp.savefig(work_dir+'perf_leftright')
pp.close()

pp.figure()
pp.imshow(CM[:,0,:,:].mean(0), origin='bottom', cmap='Greys', vmax=n_pat_test)
pp.colorbar()
pp.savefig(work_dir+'CM_mean_leftright')
pp.close()
