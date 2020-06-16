#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:03:18 2019

@author: gilsonmatthieu
"""

import numpy as np
import sklearn.linear_model as skllm
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.metrics as skm
import matplotlib.pyplot as pp



#%%
# init

work_dir = 'mnist5digits_MLR/'

grph_fmt = 'eps'


cols_gr = []
for ii in [2,1,0]:
    cols_gr += [[ii*0.3,ii*0.3,ii*0.3]]


#%% data
data_dir = 'mnist_data/'
train_data = np.load(data_dir+'train_data.npy')
test_data = np.load(data_dir+'test_data.npy')
train_labels = np.load(data_dir+'train_labels.npy')
test_labels = np.load(data_dir+'test_labels.npy')

nrows = 9 # size of images
n_digits = 5 # number of distinct digits

# create labels for each motion direction
train_labels = np.concatenate((train_labels,train_labels+n_digits))
test_labels = np.concatenate((test_labels,test_labels+n_digits))

n_cat = np.unique(train_labels).size # number of categories

# double the number of input samples (for two motion directions)
n_train = train_labels.size
n_test = test_labels.size

v_n_pat = np.array([50000],dtype=np.int)
n_n_pat = v_n_pat.size


#%%

T0 = 20
n_shift = 1

# 2 columns of receptors
M = nrows*2
M0 = nrows

T = nrows + n_shift # move all digit till end

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


# mean
mean_train = np.zeros([n_train,M])
mean_test = np.zeros([n_test,M])

# lagged covariance
cov_train = np.zeros([n_train,M,M])
cov_test = np.zeros([n_test,M,M])

# calculate for all digits/direction
for i_train in range(n_train):
    I = get_input(train_data, i_train, n_train)
    mean_train[i_train,:] = I.sum(axis=1)
    cov_train[i_train,:,:] = np.tensordot(I[:,1:],I[:,:-1],axes=(1,1)) / (T-1)

for i_test in range(n_test):
    I = get_input(test_data, i_test, n_test)
    mean_test[i_test,:] = I.sum(axis=1)
    cov_test[i_test,:,:] = np.tensordot(I[:,1:],I[:,:-1],axes=(1,1)) / (T-1)

mask_tri = np.tri(M, M, 0, dtype=np.bool)
cov_train = cov_train[:,mask_tri]
cov_test = cov_test[:,mask_tri]


#%%
# classification

i_n_pat = 0

n_rep = 10

# mean, cov, cov with m cov entries (randomly chosen) instead of m(m-1)/2
perf = np.zeros([n_rep,3])
CM = np.zeros([n_rep,3,n_cat,n_cat])

c_MLR = skppl.make_pipeline(skppc.StandardScaler(),skllm.LogisticRegression(C=10000, penalty='l2', multi_class='ovr', solver='lbfgs', max_iter=5000))


for i_rep in range(n_rep):
    print(i_rep)
    
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
    
    # train and test classifiers with subject labels    
    c_MLR.fit(mean_train[subset_train,:], train_labels[subset_train])
    perf[i_rep,0] = c_MLR.score(mean_test[subset_test,:], test_labels[subset_test])
    CM[i_rep,0,:,:] = skm.confusion_matrix(y_true=test_labels[subset_test], y_pred=c_MLR.predict(mean_test[subset_test,:]))

    c_MLR.fit(cov_train[subset_train,:], train_labels[subset_train])
    perf[i_rep,1] = c_MLR.score(cov_test[subset_test,:], test_labels[subset_test])
    CM[i_rep,1,:,:] = skm.confusion_matrix(y_true=test_labels[subset_test], y_pred=c_MLR.predict(cov_test[subset_test,:]))
    
    sub_ind_in = np.zeros([int(M*(M+1)/2)], dtype=np.bool)
    while sub_ind_in.sum()<M:
        sub_ind_in[np.random.randint(int(M*(M+1)/2))] = True
    c_MLR.fit(cov_train[subset_train,:][:,sub_ind_in], train_labels[subset_train])
    perf[i_rep,2] = c_MLR.score(cov_test[subset_test,:][:,sub_ind_in], test_labels[subset_test])
    CM[i_rep,2,:,:] = skm.confusion_matrix(y_true=test_labels[subset_test], y_pred=c_MLR.predict(cov_test[subset_test,:][:,sub_ind_in]))
    
    print(perf[i_rep,:])
        

np.save(work_dir+'perf.npy',perf)


#%%    
# plots
    
pp.figure()
pp.violinplot(perf, positions=range(3))
pp.plot([-0.5,2.5],[1/n_cat,1/n_cat],'--k')
pp.axis(ymin=0,ymax=1)
pp.xticks([0,1,2],['mean','cov','cov subsampl'])
pp.ylabel('perf')
pp.savefig(work_dir+'perf_leftright')
pp.close()

pp.figure()
pp.imshow(CM[:,0,:,:].mean(0), origin='bottom', cmap='Greys', vmax=n_pat_test/M)
pp.colorbar()
pp.savefig(work_dir+'CM_mean_leftright')
pp.close()

pp.figure()
pp.imshow(CM[:,1,:,:].mean(0), origin='bottom', cmap='Greys', vmax=n_pat_test/M)
pp.colorbar()
pp.savefig(work_dir+'CM_cov_leftright')
pp.close()

pp.figure()
pp.imshow(CM[:,2,:,:].mean(0), origin='bottom', cmap='Greys', vmax=n_pat_test/M)
pp.colorbar()
pp.savefig(work_dir+'CM_cov_subsampl_leftright')
pp.close()

