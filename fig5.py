#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:17:04 2020

@author: gilsonmatthieu
"""

import os
import numpy as np
import matplotlib.pyplot as pp


work_dir = 'fig5/'
if not os.path.exists(work_dir):
    print('create directory:', work_dir)
    os.makedirs(work_dir)

grph_fmt = 'eps'


cols = ['Reds','Blues']
cols2 = ['r','b','k']
cols3 = []
for i_T in range(3):
    cols3 += [[]]
    cols3[i_T] += [[1,0.8-i_T*0.2,0.8-i_T*0.2],[0.8-i_T*0.2,0.8-i_T*0.2,1],[0.8-i_T*0.2,0.8-i_T*0.2,0.8-i_T*0.2]]

cols4 = []
for i_T in range(3):
    cols4 += [[0.8-i_T*0.3,0.8-i_T*0.3,0.8-i_T*0.3]]


#%% network model

M = 10 # number of inputs
N = 2 # number of outputs

sparse_input = 0.1 # sparsity of input matrices

mask_offdiag = np.logical_not(np.eye(N,dtype=bool))
mask_tri = np.tri(N,N,-1,dtype=np.bool)

# simulatation duration for network activity
T0 = 5 # number of initial points to discard (memoryless process here, not needed)
v_T = np.array([10,20,30],dtype=np.int) # window sizes
n_T = v_T.size # number of window sizes to test
T = v_T[-1] # time for simulation

# simulation of network in response to (noisy) input time series
def sim_net(W_arg,B_arg,T_arg):
    noise_x = np.random.randn(M,T0+T_arg) # realization of noise to generate input
    x_tmp = np.dot(W_arg, noise_x) # graphical model
    y_tmp = np.dot(B_arg, x_tmp) # output time series
    return x_tmp[:,T0:], y_tmp[:,T0:]

# compute input/output covariance from network activity
def comp_cov_emp(ts_x_arg,ts_y_arg,T_arg):
    P0_tmp = np.tensordot(ts_x_arg[:,0:T_arg-1],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    Q0_tmp = np.tensordot(ts_y_arg[:,0:T_arg-1],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    return P0_tmp, Q0_tmp


#%% optimization for classification

eta_B = 0.05 # learning rate

n_opt_aff = 1000 # number of optimization steps
n_smooth = 5 # smoothing for performance curve
n_opt = n_opt_aff + n_smooth # number of optimization steps including discarded initial steps

n_pat = 10 # number of patterns in total
n_cat = 2 # number of categories
v_match = np.zeros([n_pat],dtype=np.int) # category for each input pattern
for i_pat in range(n_pat):
     v_match[i_pat] = int(i_pat/(n_pat/n_cat)) # even number of patterns per category

n_rep = 20 # optimization repetitions
n_samp = 100 # number of samples to test classification accuracy

# save error history and accuracy
err_hist = np.zeros([n_rep,n_T,n_opt])
acc_summary = np.zeros([n_rep,n_T])
    
# basis matrices; 1 for indices (i,k), 0 otherwise
M_ik = np.zeros([N,M,N,M])
for i in range(N):
    for k in range(M):
        M_ik[i,k,i,k] = 1
    

for i_rep in range(n_rep):

    # generate mixing matrix that determine input patterns (graphical model)
    W_pat = np.zeros([n_pat,M,M])
    for i_pat in range(n_pat):
        # input mixing matrix
        W_pat[i_pat,:,:] = 0.5 + 0.5*np.random.rand(M,M)
        W_pat[i_pat,np.random.rand(M,M)>=sparse_input] = 0
        if np.abs(np.linalg.eig(W_pat[i_pat,:,:])[0]).max()>0:
            W_pat[i_pat,:,:] *= (0.4 + 0.3*np.random.rand()) / np.abs(np.linalg.eig(W_pat[i_pat,:,:])[0]).max()

    # generate output objective patterns
    Q0_cat = np.zeros([n_cat,N,N])
    for i_cat in range(n_cat):
        # pattern category correspond to larger variance
        Q0_cat[i_cat,:,:] = np.eye(N) * 0.3
        Q0_cat[i_cat,i_cat,i_cat] = 1

    # loop over window sizes
    for i_T in range(n_T):
        print('rep/i_T:',i_rep,i_T)

        # initial conditions
        B_ini = np.random.rand(N,M) * 0.2
        B = np.array(B_ini)
                
        # optimization loop
        for i_opt in range(n_opt):
            
            # randomly choose a pattern
            i_pat = np.random.randint(n_pat)
            i_cat = v_match[i_pat]
            W = W_pat[i_pat,:,:]
            # simulate activity and calculate empirical covriances
            x_sim,y_sim = sim_net(W,B,v_T[i_T])
            P0_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,v_T[i_T])
    
            # error on output covariance
            err_Q0_sim = Q0_cat[i_cat,:,:] - Q0_sim
            err_hist[i_rep,i_T,i_opt] = np.mean(err_Q0_sim**2)
            
            # derivative of Q0 wrt B
            P0_B = np.dot(P0_sim, B.T)
            d_Q0_B = np.einsum('imkl, mj -> ijkl', M_ik, P0_B) + np.einsum('jmkl, mi -> ijkl', M_ik, P0_B)
                    
            # weight updates
            B += eta_B * np.einsum('jl, jlik -> ik', err_Q0_sim, d_Q0_B)

        B_fin = np.array(B)
        
        # test accurqacy at the end of the optimization
        for i_samp in range(n_samp):
            i_pat = int((i_samp*n_pat)/n_samp)
            i_cat = v_match[i_pat]
            W = W_pat[i_pat,:,:]
            x_sim,y_sim = sim_net(W,B_fin,v_T[i_T])
            P0_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,v_T[i_T])
            acc_summary[i_rep,i_T] += np.logical_xor(Q0_sim[0,0]-Q0_sim[1,1]>0,i_cat==1)
                
acc_summary /= n_samp


#%% result plots

# smooth error curve
smoothed_err = np.zeros(err_hist.shape)
for i_rep in range(n_rep):
    for i_T in range(n_T):
        smoothed_err[i_rep,i_T,:] = np.convolve(err_hist[i_rep,i_T,:],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]

pp.figure(figsize=[4.5,2])
pp.axes([0.2,0.3,0.7,0.6])
for i_T in range(n_T):
    pp.plot(range(n_opt_aff),smoothed_err[0,i_T,:n_opt_aff],c=cols4[i_T])
pp.xticks(np.arange(0,n_opt+1,200),fontsize=10)
pp.yticks([0,0.2,0.4],fontsize=10)
pp.axis(ymin=0,ymax=0.4)
pp.legend(['$d$='+str(v_T[0]),'$d$='+str(v_T[1]),'$d$='+str(v_T[2])],fontsize=7)
pp.ylabel('error $Q^0$',fontsize=10)
pp.xlabel('optimization steps',fontsize=10)
pp.savefig(work_dir+'err_hist.'+grph_fmt,format=grph_fmt)
pp.close()

# summary of accuracy after optimization for each window size
pp.figure(figsize=[2,2])
pp.axes([0.25,0.3,0.7,0.6])
for i_T in range(n_T):
    vp_tmp = pp.violinplot(acc_summary[:,i_T],positions=[i_T],widths=[0.5])
    for lbl_tmp in ('cbars','cmins','cmaxes'):
        vp_tmp[lbl_tmp].set_edgecolor(cols2[2])
    for p_tmp in vp_tmp['bodies']:
        p_tmp.set_facecolor(cols3[i_T][2])
        p_tmp.set_edgecolor(cols2[2])
pp.plot([-1,n_T],[0.8,0.8],'--k')
pp.xticks(range(n_T),v_T,fontsize=10)
pp.yticks([0.5,1],fontsize=10)
pp.axis(xmin=-0.4,xmax=n_T-0.6,ymin=0.5,ymax=1)
pp.xlabel('observation\nduration $d$',fontsize=10)
pp.ylabel('accuracy',fontsize=10)
pp.savefig(work_dir+'acc_summary.'+grph_fmt,format=grph_fmt)
pp.close()
