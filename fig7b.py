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


work_dir = 'fig7b/'
if not os.path.exists(work_dir):
    print('create directory:', work_dir)
    os.makedirs(work_dir)

grph_fmt = 'eps'

cols_g = []
for k in range(3,8):
    cols_g += [[0,k*0.1,0]]
cols_p = []
for k in range(3,8):
    cols_p += [[k*0.1,0,k*0.1]]
cols_g = cols_g * 20
cols_p = cols_p * 20

cols_gr = []
for k in range(10):
    cols_gr += [[k*0.05+0.25,k*0.05+0.25,k*0.05+0.25]]


#%% model parameters

M = 20 # number of inputs
N = 5 # number of outputs

n_opt_aff = 500 # number of optimization steps
n_smooth = 5 # smoothing for error
n_opt = n_opt_aff + n_smooth

eta_B = 0.01 # learning rate for afferent connections
eta_A = 0.01 # learning rate for recurrent connections

T0 = 10 # discarded simulation steps to avoid effects of initial conditions
T = 50 # window size to calculate covariances

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
    Q1_tmp = np.tensordot(ts_y_arg[:,1:T_arg],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-1)
    return P0_tmp, P1_tmp, Q0_tmp, Q1_tmp

# generator for basis matrices
def gen_ik(i,k):
    M_tmp = np.zeros([N,M])
    M_tmp[i,k] = 1
    return M_tmp

def gen_ij(i,j):
    M_tmp = np.zeros([N,N])
    M_tmp[i,j] = 1
    return M_tmp


#%% pattern parameters
    
n_pat = 10 # number of input patterns

# randomly generate input patterns
W_pat = np.zeros([n_pat,M,M])
for i_pat in range(n_pat):
    W_pat[i_pat,:,:] = np.random.rand(M,M) * 0.4/M/0.1
    W_pat[i_pat,:,:][np.random.rand(M,M)>0.1] = 0

# single objective matrix for all patterns (decorrelate spatially)     
Q0_obj = np.eye(N)*0.5 # in the paper, 0.5 is replaced by 1
Q1_obj = np.eye(N)*0.1 # in the paper, 0.1 is replaced by 0.2


#%% optimization to check stability

# initial conditions
B = np.random.rand(N,M) * 0.1
A = np.random.rand(N,N) * 0.05

# record history
err_hist = np.zeros([n_opt,2])
Pearson_hist = np.zeros([n_opt,2])
B_hist = np.zeros([n_opt,N,M])
A_hist = np.zeros([n_opt,N,N])

# optimization loop
for i_opt in range(n_opt):
    # randomly choose an input pattern
    i_pat = np.random.randint(n_pat)
    W = W_pat[i_pat,:,:]
    
    # simulate activity and calculate empirical covriances
    x_sim,y_sim = sim_net(W,A,B,T)
    P0_sim, P1_sim, Q0_sim, Q1_sim = comp_cov_emp(x_sim,y_sim,T)
    
    # error and Pearson on output cov
    delta_Q0_sim = Q0_obj - Q0_sim
    delta_Q1_sim = Q1_obj - Q1_sim
    err_hist[i_opt,:] = np.linalg.norm(delta_Q0_sim)/np.linalg.norm(Q0_obj),np.linalg.norm(delta_Q1_sim)/np.linalg.norm(Q1_obj)
    Pearson_hist[i_opt,:] = stt.pearsonr(Q0_obj.flatten(),Q0_sim.flatten())[0],stt.pearsonr(Q1_obj.flatten(),Q1_sim.flatten())[0]
    
    # deriv Q0 wrt B
    d_Q0_B = np.zeros([N,N,N,M])
    d_Q1_B = np.zeros([N,N,N,M])
    for i in range(N):
        for k in range(M):
            R_tmp = np.dot(gen_ik(i,k),np.dot(P0_sim,B.T)) + np.dot(A,np.dot(gen_ik(i,k),np.dot(P1_sim.T,B.T))) + \
                    np.dot(A,np.dot(B,np.dot(P1_sim.T,gen_ik(i,k).T)))
            d_Q0_B[:,:,i,k] = spl.solve_discrete_lyapunov(A,R_tmp+R_tmp.T)
            R_tmp = np.dot(gen_ik(i,k),np.dot(P1_sim,B.T)) + np.dot(B,np.dot(P1_sim,gen_ik(i,k).T)) + \
                    np.dot(A,np.dot(gen_ik(i,k),np.dot(P0_sim,B.T))) + np.dot(A,np.dot(B,np.dot(P0_sim,gen_ik(i,k).T))) + \
                    np.dot(A,np.dot(A,np.dot(gen_ik(i,k),np.dot(P1_sim.T,B.T)))) + np.dot(A,np.dot(A,np.dot(B,np.dot(P1_sim.T,gen_ik(i,k).T))))
            d_Q1_B[:,:,i,k] = spl.solve_discrete_lyapunov(A,R_tmp)
    # deriv Q0 wrt A
    d_Q0_A = np.zeros([N,N,N,N])
    d_Q1_A = np.zeros([N,N,N,N])
    for i in range(N):
        for j in range(N):
            R_tmp = np.dot(gen_ij(i,j),np.dot(Q0_sim,A.T)) + np.dot(gen_ij(i,j),np.dot(B,np.dot(P1_sim.T,B.T)))
            d_Q0_A[:,:,i,j] = spl.solve_discrete_lyapunov(A,R_tmp+R_tmp.T)
            R_tmp = np.dot(gen_ij(i,j),np.dot(Q1_sim,A.T)) + np.dot(A,np.dot(Q1_sim,gen_ij(i,j).T)) + \
                    np.dot(gen_ij(i,j),np.dot(A,np.dot(B,np.dot(P1_sim.T,B.T)))) + np.dot(A,np.dot(gen_ij(i,j),np.dot(B,np.dot(P1_sim.T,B.T)))) + \
                    np.dot(gen_ij(i,j),np.dot(B,np.dot(P0_sim,B.T)))
            d_Q1_A[:,:,i,j] = spl.solve_discrete_lyapunov(A,R_tmp)
    # weight update
    B += eta_B * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_B[:,:,:,:].reshape([-1,N,M]),axes=(0,0)) + \
                 np.tensordot(delta_Q1_sim.reshape(-1),d_Q1_B[:,:,:,:].reshape([-1,N,M]),axes=(0,0))
    A += eta_A * np.tensordot(delta_Q0_sim.reshape(-1),d_Q0_A[:,:,:,:].reshape([-1,N,N]),axes=(0,0)) + \
                 np.tensordot(delta_Q1_sim.reshape(-1),d_Q1_A[:,:,:,:].reshape([-1,N,N]),axes=(0,0))
    
    # record weights for plots
    B_hist[i_opt,:,:] = B
    A_hist[i_opt,:,:] = A

        
#%% result plots

# smoothed error and Pearson correlation over time
smoothed_err = np.zeros([n_opt,2])
smoothed_Pearson = np.zeros([n_opt,2])
smoothed_err[:,0] = np.convolve(err_hist[:,0],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_err[:,1] = np.convolve(err_hist[:,1],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_Pearson[:,0] = np.convolve(Pearson_hist[:,0],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_Pearson[:,1] = np.convolve(Pearson_hist[:,1],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]

pp.figure(figsize=[2.5,2.5])
pp.axes([0.25,0.65,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,0],color='k',lw=2)
pp.xticks(np.arange(0,n_opt+1,500),[],fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.axis(ymin=0)
pp.ylabel('norm error $Q^0$',fontsize=10)
pp.axes([0.25,0.2,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,1],color='k',lw=2)
pp.xticks(np.arange(0,n_opt+1,500),fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.axis(ymin=0)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('norm error $Q^1$',fontsize=10)
pp.savefig(work_dir+'err_hist.'+grph_fmt,format=grph_fmt)
pp.close()

pp.figure(figsize=[2.5,2.5])
pp.axes([0.25,0.65,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_Pearson[:n_opt_aff,0],color='k',lw=2)
pp.xticks(np.arange(0,n_opt+1,500),[],fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.axis(ymin=0,ymax=1)
pp.ylabel('Pearson corr $Q^0$',fontsize=10)
pp.axes([0.25,0.2,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_Pearson[:n_opt_aff,1],color='k',lw=2)
pp.xticks(np.arange(0,n_opt+1,500),fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.axis(ymin=0,ymax=1)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('Pearson corr $Q^1$',fontsize=10)
pp.savefig(work_dir+'Pearson_hist.'+grph_fmt,format=grph_fmt)
pp.close()


# plot weight traces
pp.figure(figsize=[3.5,2.5])
pp.axes([0.25,0.65,0.7,0.3])
for i in range(N):
    for j in range(N):
        if np.random.rand()<1:
            pp.plot(range(n_opt_aff),B_hist[:n_opt_aff,i,j],c=cols_p[i*N+j])
pp.xticks(range(0,n_opt_aff+1,500),[],fontsize=10)
pp.yticks([-0.5,0,0.5],fontsize=10)
pp.ylabel('recurrent\nweights $A$',fontsize=10)
pp.axes([0.25,0.2,0.7,0.35])
for i in range(N):
    for k in range(M):
        if np.random.rand()<1:
            pp.plot(range(n_opt_aff),B_hist[:n_opt_aff,i,k],c=cols_g[i*M+k])
pp.xticks(range(0,n_opt_aff+1,500),fontsize=10)
pp.yticks([-0.5,0,0.5],fontsize=10)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('afferent\nweights $B$',fontsize=10)
pp.savefig(work_dir+'BA_hist.'+grph_fmt,format=grph_fmt)
pp.close()