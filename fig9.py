#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:17:04 2020

@author: gilsonmatthieu
"""

import os
import numpy as np
import scipy.linalg as spl
import scipy.stats as stt
import matplotlib.pyplot as pp


work_dir = 'fig9/'
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

mask_offdiag = np.logical_not(np.eye(N,dtype=bool)) # mask for off-diagonal elements

# simulate network activity without noise to get steady state
T0 = 20 # simulation steps to discard (avoiding effects of initial conditions)
v_T = np.array([20,60,100],dtype=np.int) # window sizes to caompute covariances
n_T = v_T.size # number of window sizes to test
T = v_T[-1] # duration for simulation

# simulation of network activity in response to (noisy) input time series
def sim_net(W_arg,A_arg,B_arg,T_arg):
    noise_x = np.random.randn(M,T0+T_arg)
    x_tmp = np.copy(noise_x)
    x_tmp[:,1:] += np.dot(W_arg,x_tmp[:,:-1])
    y_tmp = np.dot(B_arg,x_tmp)
    y_tmp[:,1:] += np.dot(A_arg,y_tmp[:,:-1])
    return x_tmp[:,T0:], y_tmp[:,T0:]

# calculation of input/output covariance from network activity
def comp_cov_emp(ts_x_arg,ts_y_arg,T_arg):
    P0_tmp = np.tensordot(ts_x_arg[:,0:T_arg-1],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    P1_tmp = np.tensordot(ts_x_arg[:,1:T_arg],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    Q0_tmp = np.tensordot(ts_y_arg[:,0:T_arg-1],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    Q1_tmp = np.tensordot(ts_y_arg[:,1:T_arg],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    return P0_tmp, P1_tmp, Q0_tmp, Q1_tmp

# basis matrices; 1 for indices (i,k) or (i,j), 0 otherwise
M_ik = np.zeros([N,M,N,M])
for i in range(N):
    for k in range(M):
        M_ik[i,k,i,k] = 1
        
M_ij = np.zeros([N,N,N,N])
for i in range(N):
    for j in range(N):
        M_ij[i,j,i,j] = 1
        

#%% optimization for classification

eta_B = 0.02 # learning rate for afferent weights
eta_A = 0.02 # learning rate for recurrent weights

n_opt_aff = 1000 # number of optimization steps
n_smooth = 5 # smoothing for performance curve
n_opt = n_opt_aff + n_smooth # number of optimization steps including discarded initial steps

n_pat = 6 # number of input patterns in total
n_cat = 2 # number of categories
v_match = np.zeros([n_pat],dtype=np.int) # category for each input pattern
for i_pat in range(n_pat):
     v_match[i_pat] = int(i_pat/(n_pat/n_cat)) # even number of patterns per category

n_rep = 20 # repetition of same experiment
n_samp = 100 # number of samples to test classification accuracy

# save error history and accuracy
err_hist = np.zeros([n_rep,n_T,n_opt])
acc_summary = np.zeros([n_rep,n_T])
    
i_rep = 0
while i_rep<n_rep:

    # randomly generate input patterns
    W_pat = np.zeros([n_pat,M,M])
    for i_pat in range(n_pat):
        # generate antisymmetric matrix
        antisym_W = np.zeros([M,M])
        for i in range(M):
            for j in range(i):
                if np.random.rand()<0.3:
                    antisym_W[j,i] = (0.5 + 0.5*np.random.rand()) * (1-2*np.random.randint(2))
                    antisym_W[i,j] = -antisym_W[j,i]
        # input mixing matrix W to obtained spatially uncorrelated inputs
        W_pat[i_pat,:,:] = spl.expm(-np.eye(M)/2+antisym_W)

    # create output objective patterns        
    Q0_cat = np.zeros([n_cat,N,N])
    for i_cat in range(n_cat):
        # pattern category corresponds to larger variance
        Q0_cat[i_cat,:,:] = np.eye(N) * 0.3
        Q0_cat[i_cat,i_cat,i_cat] = 1

    cnt_bad_sim = 0
    discard_sim = False

    # loop over window sizes
    for i_T in range(n_T):
        print('rep/i_T:',i_rep,i_T)
        good_sim = False
        while not good_sim and not discard_sim:
            try:
                # initial conditions
                B_ini = np.random.rand(N,M) * 0.3
                B = np.array(B_ini)
                A_ini = np.random.rand(N,N) * 0.01
                A = np.array(A_ini)
                        
                # optimization loop
                for i_opt in range(n_opt):
                    # randomly choose input pattern
                    i_pat = np.random.randint(n_pat)
                    i_cat = v_match[i_pat]
                    W = W_pat[i_pat,:,:]
                    
                    # simulate activity and calculate empirical covariances
                    x_sim,y_sim = sim_net(W,A,B,v_T[i_T])
                    P0_sim, P1_sim, Q0_sim, Q1_sim = comp_cov_emp(x_sim,y_sim,v_T[i_T])
            
                    # error on output covariance Q0
                    delta_Q0_sim = Q0_cat[i_cat,:,:] - Q0_sim
                    err_hist[i_rep,i_T,i_opt] = np.mean(delta_Q0_sim**2)

                    # derivivative of Q0 wrt B
                    d_Q0_B = np.zeros([N,N,N,M])
                    for i in range(N):
                        for k in range(M):
                            R_tmp = np.dot(M_ik[i,k],np.dot(P0_sim,B.T)) + np.dot(A,np.dot(M_ik[i,k],np.dot(P1_sim.T,B.T))) + \
                                    np.dot(A,np.dot(B,np.dot(P1_sim.T,M_ik[i,k].T)))
                            d_Q0_B[:,:,i,k] = spl.solve_discrete_lyapunov(A,R_tmp+R_tmp.T)
                    # derivative of Q0 wrt A
                    d_Q0_A = np.zeros([N,N,N,N])
                    for i in range(N):
                        for j in range(N):
                            R_tmp = np.dot(M_ij[i,j],np.dot(Q0_sim,A.T)) + np.dot(M_ij[i,j],np.dot(B,np.dot(P1_sim.T,B.T)))
                            d_Q0_A[:,:,i,j] = spl.solve_discrete_lyapunov(A,R_tmp+R_tmp.T)
                    # weight update
                    B += eta_B * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_B[:,:,:,:].reshape([-1,N,M]),axes=(0,0))
                    A += eta_A * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_A[:,:,:,:].reshape([-1,N,N]),axes=(0,0))                
            
                B_fin = np.array(B)
                A_fin = np.array(A)
                good_sim = True
            except:
                print('bad sim')
                cnt_bad_sim += 1
                if cnt_bad_sim>=3:
                    discard_sim = True
        
        # test accuracy at the end of the optimization
        if not discard_sim:
            for i_samp in range(n_samp):
                i_pat = int((i_samp*n_pat)/n_samp)
                i_cat = v_match[i_pat]
                W = W_pat[i_pat,:,:]
                x_sim,y_sim = sim_net(W,A_fin,B_fin,v_T[i_T])
                P0_sim, P1_sim, Q0_sim, Q1_sim = comp_cov_emp(x_sim,y_sim,v_T[i_T])
                acc_summary[i_rep,i_T] += np.logical_xor(Q0_sim[0,0]-Q0_sim[1,1]>0,i_cat==1)
      
    if not discard_sim:
        i_rep += 1

acc_summary /= n_samp


#%% result plots

# smooth error curve
smoothed_err0 = np.zeros(err_hist.shape)
for i_rep in range(n_rep):
    for i_T in range(n_T):
        smoothed_err0[i_rep,i_T,:] = np.convolve(err_hist[i_rep,i_T,:],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]

pp.figure(figsize=[4.5,2])
pp.axes([0.2,0.3,0.7,0.6])
for i_T in range(n_T):
    pp.plot(range(n_opt_aff),smoothed_err0[0,i_T,:n_opt_aff],c=cols4[i_T])
pp.xticks(np.arange(0,n_opt+1,200),fontsize=10)
pp.yticks([0,0.1,0.2,0.3],fontsize=10)
pp.axis(ymin=0,ymax=0.35)
pp.legend(['$d$='+str(v_T[0]),'$d$='+str(v_T[1]),'$d$='+str(v_T[2])],fontsize=7)
pp.ylabel('error $Q^0$',fontsize=10)
pp.xlabel('optimization steps',fontsize=10)
pp.savefig(work_dir+'err_hist.'+grph_fmt,format=grph_fmt)
pp.close()


# summary of accuracy after optimization for each window size
pp.figure(figsize=[2,2])
pp.axes([0.3,0.3,0.65,0.6])
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

