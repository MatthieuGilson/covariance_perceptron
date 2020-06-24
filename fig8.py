#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:17:04 2020

@author: gilsonmatthieu
"""

import os
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as pp


work_dir = 'fig8/'
if not os.path.exists(work_dir):
    print('create directory:', work_dir)
    os.makedirs(work_dir)

grph_fmt = 'eps'


#%% network model

M = 20 # number of inputs
N = 5 # number of outputs

n_opt_aff = 500 # number of optimization steps
n_smooth = 5 # smoothing for error
n_opt = n_opt_aff + n_smooth

eta_B = 0.01 # learning rate for afferent connections
eta_A = 0.01 # learning rate for recurrent connections

T0 = 10 # discarded simulation steps to avoid effects of initial conditions
T = 50 # window size to calculate covariances
#T0 = 50
#T = 100

# simulation of network activity in response to (noisy) input time series
def sim_net(W_arg,A_arg,B_arg,T_arg):
    noise_x = np.random.randn(M,T0+T_arg)
    x_tmp = np.copy(noise_x)
    x_tmp[:,1:] += np.dot(W_arg,x_tmp[:,:-1])
    y_tmp = np.dot(B_arg,x_tmp)
    y_tmp[:,1:] += np.dot(A_arg,y_tmp[:,:-1])
    return x_tmp[:,T0:], y_tmp[:,T0:]

# calculation of input/output covariances
def comp_cov_emp(ts_x_arg,ts_y_arg,T_arg):
    P0_tmp = np.tensordot(ts_x_arg[:,0:T_arg-1],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    P1_tmp = np.tensordot(ts_x_arg[:,1:T_arg],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    Q0_tmp = np.tensordot(ts_y_arg[:,0:T_arg-1],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    return P0_tmp, P1_tmp, Q0_tmp

# basis matrices; 1 for indices (i,k) or (i,j), 0 otherwise
M_ik = np.zeros([N,M,N,M])
for i in range(N):
    for k in range(M):
        M_ik[i,k,i,k] = 1
        
M_ij = np.zeros([N,N,N,N])
for i in range(N):
    for j in range(N):
        M_ij[i,j,i,j] = 1

# masks
mask_diag = np.eye(N,dtype=np.bool)
mask_offdiag = np.logical_not(mask_diag)

mask_A = np.random.rand(N,N)<0.3
mask_notA = np.logical_not(mask_A)

mask_B = np.random.rand(N,M)<0.3
mask_notB = np.logical_not(mask_B)

mask_Q0 = np.logical_or(mask_A, np.eye(N, dtype=np.bool))
mask_notQ0 = np.logical_not(mask_Q0)


#%% pattern parameters

n_pat = 10 # number of input patterns

# randomly generate input patterns with diagonal P0 = W_pat W_pat^T matrix
W_pat = np.zeros([n_pat,M,M])
for i_pat in range(n_pat):
    # gen antisymmetric matrix
    antisym_W = np.zeros([M,M])
    for i in range(M):
        for j in range(i):
            if not i==j and np.random.rand()<0.3:
                antisym_W[j,i] = (0.5 + 0.5*np.random.rand()) * (1-2*np.random.randint(2))
                antisym_W[i,j] = -antisym_W[j,i]
    # input mixing matrix W
    W_pat[i_pat,:,:] = spl.expm(-np.eye(M)/4+antisym_W)

# single objective matrix for all patterns
Q0_obj_sqrt = 0.2 + 0.3 * np.random.rand(N,N)
Q0_obj_sqrt *= (np.random.randint(2, size=[N,N])-0.5) * 2
Q0_obj_sqrt[np.random.rand(N,N)>0.3] = 0
Q0_obj_sqrt[mask_diag] = 0
Q0_obj_sqrt += 0.8 * np.eye(N)
Q0_obj = np.dot(Q0_obj_sqrt, Q0_obj_sqrt.T)


#%% optimization to compare optimization methods

# initial conditions
Bf = np.random.rand(N,M) * 0.1
Af = np.random.rand(N,N) * 0.05
Af[mask_notA] = 0
Ba = np.copy(Bf)
Aa = np.copy(Af)
Bal = np.copy(Bf)
Aal = np.copy(Af)

# save history
err_hist = np.zeros([n_opt,3])

for i_opt in range(n_opt):
    i_pat = np.random.randint(n_pat)

    W = W_pat[i_pat,:,:]
    
    # full gradient
    x_sim,y_sim = sim_net(W,Af,Bf,T)
    P0_sim, P1_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,T)
    
    delta_Q0_sim = Q0_obj - Q0_sim
    err_hist[i_opt,0] = np.linalg.norm(delta_Q0_sim)/np.linalg.norm(Q0_obj)

    d_Q0_B = np.zeros([N,N,N,M])
    for i in range(N):
        for k in range(M):
            R_tmp = np.dot(M_ik[i,k],np.dot(P0_sim,Bf.T)) + np.dot(Af,np.dot(M_ik[i,k],np.dot(P1_sim.T,Bf.T))) + \
                    np.dot(Af,np.dot(Bf,np.dot(P1_sim.T,M_ik[i,k].T)))
            d_Q0_B[:,:,i,k] = spl.solve_discrete_lyapunov(Af,R_tmp+R_tmp.T)

    d_Q0_A = np.zeros([N,N,N,N])
    for i in range(N):
        for j in range(N):
            R_tmp = np.dot(M_ij[i,j],np.dot(Q0_sim,Af.T)) + np.dot(M_ij[i,j],np.dot(Bf,np.dot(P1_sim.T,Bf.T)))
            d_Q0_A[:,:,i,j] = spl.solve_discrete_lyapunov(Af,R_tmp+R_tmp.T)

    Bf += eta_B * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_B[:,:,:,:].reshape([-1,N,M]),axes=(0,0))
    Bf[mask_notB] = 0
    Af += eta_A * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_A[:,:,:,:].reshape([-1,N,N]),axes=(0,0))
    Af[mask_notA] = 0
                    
    # gradient with only 0 order
    x_sim,y_sim = sim_net(W,Aa,Ba,T)
    P0_sim, P1_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,T)
    
    delta_Q0_sim = Q0_obj - Q0_sim
    err_hist[i_opt,1] = np.linalg.norm(delta_Q0_sim)/np.linalg.norm(Q0_obj)
    
    d_Q0_B = np.zeros([N,N,N,M])
    for i in range(N):
        for k in range(M):
            R_tmp = np.dot(M_ik[i,k],np.dot(P0_sim,Bf.T)) + np.dot(Af,np.dot(M_ik[i,k],np.dot(P1_sim.T,Bf.T))) + \
                    np.dot(Af,np.dot(Bf,np.dot(P1_sim.T,M_ik[i,k].T)))
            d_Q0_B[:,:,i,k] = R_tmp + R_tmp.T

    d_Q0_A = np.zeros([N,N,N,N])
    for i in range(N):
        for j in range(N):
            R_tmp = np.dot(M_ij[i,j],np.dot(Q0_sim,Aa.T)) + np.dot(M_ij[i,j],np.dot(Ba,np.dot(P1_sim.T,Ba.T)))
            d_Q0_A[:,:,i,j] = R_tmp + R_tmp.T

    Ba += eta_B * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_B[:,:,:,:].reshape([-1,N,M]),axes=(0,0))
    Ba[mask_notB] = 0
    Aa += eta_A * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_A[:,:,:,:].reshape([-1,N,N]),axes=(0,0))
    Aa[mask_notA] = 0

    # neighbor gradient with 0th order
    x_sim,y_sim = sim_net(W,Aal,Bal,T)
    P0_sim, P1_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,T)
    
    delta_Q0_sim = Q0_obj - Q0_sim
    err_hist[i_opt,2] = np.linalg.norm(delta_Q0_sim)/np.linalg.norm(Q0_obj)
    
    delta_Q0_sim[mask_notQ0] = 0

    R_tmp = np.dot(P0_sim,Bal.T) + np.dot(P1_sim,np.dot(Bal.T,Aal.T))
    d_Q0_B = np.einsum('imkl, mj -> ijkl', M_ik, R_tmp) + np.einsum('jmkl, mi -> ijkl', M_ik, R_tmp)

    R_tmp = Aal.T + np.dot(Bal,np.dot(P1_sim,Bal.T))
    d_Q0_A = np.einsum('imkl, mj -> ijkl', M_ij, R_tmp) + np.einsum('jmkl, mi -> ijkl', M_ij, R_tmp)

    Bal += eta_B * np.einsum('jl, jlik -> ik', delta_Q0_sim, d_Q0_B)
    Bal[mask_notB] = 0
    Aal += eta_A * np.einsum('jl, jlik -> ik', delta_Q0_sim, d_Q0_A)
    Aal[mask_notA] = 0
                        

print('density Q0:', mask_Q0.sum()/N**2)


#%% result plots

# smoothed error and Pearson
smoothed_err = np.zeros([n_opt,3]) # last index = optimization method: full, approx, approx+neighbor
smoothed_err[:,0] = np.convolve(err_hist[:,0],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_err[:,1] = np.convolve(err_hist[:,1],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_err[:,2] = np.convolve(err_hist[:,2],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,0],color='k')
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,1],color='r')
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,2],color='m')
pp.xticks(np.arange(0,n_opt+1,500),fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.legend(['full','approx','loc+approx'], fontsize=7)
pp.axis(ymin=0)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('normalized error $Q^0$',fontsize=10)
pp.savefig(work_dir+'err_hist.'+grph_fmt,format=grph_fmt)
pp.close()


# objective covariance matrix Q0

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Q0_obj, vmin=-1, vmax=1, cmap='jet')
pp.colorbar()
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Q0_obj.'+grph_fmt,format=grph_fmt)
pp.close()


# weight and output covariance matrices for full optimization

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Af, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Af.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Bf, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Bf.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Q0_sim, vmin=-1, vmax=1, cmap='jet')
pp.colorbar()
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Q0_sim_f.'+grph_fmt,format=grph_fmt)
pp.close()


# weight and output covariance matrices for computational approximation

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Aa, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Aa.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Ba, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Ba.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Q0_sim, vmin=-1, vmax=1, cmap='jet')
pp.colorbar()
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Q0_sim_a.'+grph_fmt,format=grph_fmt)
pp.close()


# weight and output covariance matrices for computational + local approximation

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Aal, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Aal.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Bal, vmin=-0.1, vmax=0.1, cmap='bwr')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Bal.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(Q0_sim, vmin=-1, vmax=1, cmap='jet')
pp.colorbar()
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('source index',fontsize=10)
pp.ylabel('target index',fontsize=10)
pp.savefig(work_dir+'Q0_sim_al.'+grph_fmt,format=grph_fmt)
pp.close()            


# mask matrices
            
pp.figure(figsize=[2,2])
pp.axes([0.3,0.25,0.65,0.7])
pp.imshow(mask_A, vmin=0, vmax=1, cmap='binary', origin='bottom')
pp.axis(xmin=-0.5,xmax=N-0.5,ymin=-0.5,ymax=N-0.5)
pp.xticks(np.arange(N), np.arange(N)+1, fontsize=10)
pp.yticks(np.arange(N), np.arange(N)+1, fontsize=10)
pp.xlabel('source index $j$',fontsize=10)
pp.ylabel('target index $i$',fontsize=10)
pp.savefig(work_dir+'mask_A.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[3.5,2])
pp.axes([0.12,0.27,0.85,0.65])
pp.imshow(mask_B, vmin=0, vmax=1, aspect='auto', cmap='binary', origin='bottom')
pp.axis(xmin=-0.5,xmax=M-0.5,ymin=-0.5,ymax=N-0.5)
pp.xticks([0,9,19], [1,10,20], fontsize=10)
pp.yticks(np.arange(N), np.arange(N)+1, fontsize=10)
pp.xlabel('source index $k$',fontsize=10)
pp.ylabel('target index $i$',fontsize=10)
pp.savefig(work_dir+'mask_B.'+grph_fmt,format=grph_fmt)
pp.close()

