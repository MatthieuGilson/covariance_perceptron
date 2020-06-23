#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:17:04 2020

@author: gilsonmatthieu
"""

import os
import numpy as np
import scipy.stats as stt
import matplotlib.pyplot as pp


work_dir = 'fig3/'
if not os.path.exists(work_dir):
    print('create directory:', work_dir)
    os.makedirs(work_dir)

grph_fmt = 'eps'

cols = ['Reds','Blues']
cols2 = ['r','b']
cols3 = [[1,0.8,0.8],[0.8,0.8,1]]
cols_g = []
for k in range(3,8):
    cols_g += [[0,k*0.1,0]]
cols_g = cols_g * 10


#%% model parameters

M = 10 # number of inputs
N = 2 # number of outputs

n_opt_aff = 1000 # number of optimization steps
n_smooth = 5 # smoothing for performance curve
n_opt = n_opt_aff + n_smooth # number of optimization steps including discarded initial steps
eta_B = 0.01 # learning rate

eps_noise = 0.3 # artificial noise on input time series

classif_type = True # toggle for classification based on variance difference or cross-covariance

n_pat = 10 # number of input patterns in total
n_cat = 2 # number of categories
v_match = np.zeros([n_pat], dtype=np.int) # vector of category for each input pattern
for i_pat in range(n_pat):
     v_match[i_pat] = int(i_pat/(n_pat/n_cat))


#%% simulation

mask_tri = np.tri(M,M,-1,dtype=np.bool)     

# generation of input patterns
P0_pat = np.zeros([n_pat,M,M])
for i_pat in range(n_pat):
    P0_pat[i_pat,:,:] = np.eye(M)
    P0_pat[i_pat,:,:] += np.array(np.random.rand(M,M)<0.1,dtype=np.float)
    P0_pat[i_pat,mask_tri] = 0
    P0_pat[i_pat,:,:] = 0.5*(P0_pat[i_pat,:,:]+P0_pat[i_pat,:,:].T)

# function to generate a noisy version of a given input pattern
def noisy_pat(P_arg):
    P_tmp = np.array(P_arg)
    P_tmp += eps_noise * np.random.rand(M,M)
    P_tmp = 0.5*(P_tmp+P_tmp.T)
    return P_tmp

# objective matrices for each category
Q0_cat = np.zeros([n_cat,N,N])
for i_cat in range(n_cat):
    if classif_type: # discriminate by variance (and decorrelate neurons)
        Q0_cat[i_cat,:,:] = np.eye(N) * 0.3
        Q0_cat[i_cat,i_cat,i_cat] = 1
    else: # 1 category correspond to positive cross-correlations and other to none
        Q0_cat[i_cat,:,:] = np.eye(N)
        if i_cat==0:
            Q0_cat[i_cat,0,1] = 0.8
            Q0_cat[i_cat,1,0] = 0.8

# basis matrices; 1 for indices (i,k), 0 otherwise
M_ik = np.zeros([N,M,N,M])
for i in range(N):
    for k in range(M):
        M_ik[i,k,i,k] = 1

# initial conditions for the afferent weights
B_ini = np.random.rand(N,M) * 0.3
B = np.copy(B_ini)

# record history of error, performance
B_hist = np.zeros([n_opt,N,M])
err_hist = np.zeros([n_opt,N,N])
Pearson_hist = np.zeros([n_opt])

# test classification on random samples
n_samp = 100 # number of samples to test
def test_classif(B_arg):
    acc_tmp = 0 # classification accuracy
    for i_samp in range(n_samp):
        # sample over all input patterns
        i_pat = int((i_samp*n_pat)/n_samp)
        i_cat = v_match[i_pat]
        # calculate the output covariance for the noisy input pattern
        P_sim = noisy_pat(P0_pat[i_pat,:,:])
        Q_sim = np.dot(B_arg,np.dot(P_sim,B_arg.T))
        # classification scheme
        if classif_type==True: # based on variance difference
            acc_tmp += int(np.logical_xor(Q_sim[0,0]-Q_sim[1,1]>0,i_cat==1))
        else: # based on cross-variance (threshold at 0.4, midpoint of objective covariances)
            acc_tmp += int(np.logical_xor(Q_sim[0,1]>0.4,i_cat==1))
    return acc_tmp / n_samp

T_acc = 50
acc_hist = np.zeros([int(n_opt/T_acc)+1])

# optimization loop
for i_opt in range(n_opt):
    # test classification accuracy (over input samples)
    if np.mod(i_opt, T_acc):
        acc_hist[int(i_opt/T_acc)] = test_classif(B)

    # randomly choose an input pattern and get its category
    i_pat = np.random.randint(n_pat)
    i_cat = v_match[i_pat]

    # generate a noisy version of the chosen input pattern
    P_sim = noisy_pat(P0_pat[i_pat,:,:])
    
    # output covariance pattern (analytically simulating the network dynamics)
    # corresponds to the equation Q0 = B P0 B^T
    Q_sim = np.dot(B,np.dot(P_sim,B.T))
    
    # error on output covariance
    err_sim = Q0_cat[i_cat,:,:] - Q_sim

    # record error and Pearson between desired and actual output covariance patterns
    err_hist[i_opt,:,:] = err_sim**2
    Pearson_hist[i_opt] = stt.pearsonr(Q0_cat[i_cat,:,:].flatten(),Q_sim.flatten())[0]

    # record weight evolution
    B_hist[i_opt,:,:] = B
    
    # update the afferent weights B using the derivative of Q0 with respect to B
    # corresponds to U P0 B^T + B P0 U^T with U=U^ik for each weight B_ik
    d_Q0_B = np.zeros([N,N,N,M])
    for i in range(N):
        for k in range(M):
            d_Q0_B[:,:,i,k] = np.dot(M_ik[i,k], np.dot(P_sim,B.T))
            d_Q0_B[:,:,i,k] += d_Q0_B[:,:,i,k].T
    # weight update for B_ik is sum of all elements of the matrix obtained by the element-wise multiplication between 
    # the covariance difference matrix err_sim and d_Q0_B with last indices i and k
    B += eta_B * np.tensordot(err_sim.reshape(-1), d_Q0_B[:,:,:,:].reshape([-1,N,M]), axes=(0,0))
    
# save final weight matrix
B_fin = np.copy(B)


#%% result plots


# evolution of weights
pp.figure(figsize=[4.5,3])
pp.axes([0.2,0.2,0.7,0.7])
for i in range(N):
    for k in range(M):
        pp.plot(range(n_opt_aff),B_hist[:n_opt_aff,i,k],c=cols_g[i*M+k])
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('afferent weights $B$',fontsize=10)
pp.savefig(work_dir+'B_hist.'+grph_fmt,format=grph_fmt)
pp.close()


# evolution of smoothed error
smoothed_err = np.zeros(err_hist.shape)
for i in range(N):
    for j in range(N):
        smoothed_err[:,i,j] = np.convolve(err_hist[:,i,j],np.ones([n_smooth])/n_smooth,'full')[-n_opt:]
smoothed_Pearson = np.convolve(Pearson_hist,np.ones([n_smooth])/n_smooth,'full')[-n_opt:]

pp.figure(figsize=[4.5,3])
pp.axes([0.2,0.65,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,:,:].reshape([n_opt_aff,-1]),c=[0.5,0.5,0.5],lw=1,alpha=0.5)
pp.plot(range(n_opt_aff),smoothed_err[:n_opt_aff,:,:].mean(axis=(1,2)),c='k',lw=2)
pp.xticks(np.arange(0,n_opt,200),[],fontsize=10)
pp.yticks(fontsize=10)
pp.axis(ymin=0)
pp.ylabel('error',fontsize=10)
pp.axes([0.2,0.2,0.7,0.3])
pp.plot(range(n_opt_aff),smoothed_Pearson[:n_opt_aff],c='k')
pp.xticks(np.arange(0,n_opt+1,200),fontsize=10)
pp.axis(ymax=1)
pp.yticks([0,0.5,1],fontsize=10)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('Pearson correlation',fontsize=10)
pp.savefig(work_dir+'err_hist.'+grph_fmt,format=grph_fmt)
pp.close()

# evolution of accuracy over optimization steps
pp.figure(figsize=[3,1.5])
pp.axes([0.25,0.3,0.7,0.6])
pp.plot(np.arange(0,n_opt,T_acc), acc_hist)
pp.plot([0,n_opt],[0.8,0.8],'--k')
pp.xticks(fontsize=10)
pp.yticks(fontsize=10)
pp.axis(ymin=0.4,ymax=1)
pp.xlabel('optimization steps',fontsize=10)
pp.ylabel('classification\naccuracy',fontsize=10)
pp.savefig(work_dir+'acc_hist.'+grph_fmt,format=grph_fmt)
pp.close()

# matrix plots of all input patterns, as well as initial and final images of input patterns
for i_pat in range(n_pat):
    i_cat = v_match[i_pat]
    
    # input patterns
    pp.figure(figsize=[1.5,1.5])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.imshow(P0_pat[i_pat,:,:],origin='bottom',interpolation='nearest',vmin=0,vmax=1,cmap=cols[i_cat])
    pp.colorbar()
    pp.xticks(np.arange(M),np.arange(M)+1,fontsize=10)
    pp.yticks(np.arange(M),np.arange(M)+1,fontsize=10)
    pp.xticks(np.arange(M),[],fontsize=10)
    pp.yticks(np.arange(M),[],fontsize=10)
    pp.savefig(work_dir+'P0_pat%d'%i_pat+'.'+grph_fmt,format=grph_fmt)
    pp.close()
    
    # initial pattern
    Q_sim = np.dot(B_ini,np.dot(P0_pat[i_pat,:,:],B_ini.T))
    
    pp.figure(figsize=[1.5,1.5])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.imshow(Q_sim,origin='bottom',interpolation='nearest',vmin=0,vmax=1,cmap=cols[i_cat])
    pp.colorbar()
    pp.xticks(np.arange(N),[],fontsize=10)
    pp.yticks(np.arange(N),[],fontsize=10)
    pp.savefig(work_dir+'ini_Q0_pat%d'%i_pat+'.'+grph_fmt,format=grph_fmt)
    pp.close()

    # final pattern
    Q_sim = np.dot(B_fin,np.dot(P0_pat[i_pat,:,:],B_fin.T))
    
    pp.figure(figsize=[1.5,1.5])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.imshow(Q_sim,origin='bottom',interpolation='nearest',vmin=0,vmax=1,cmap=cols[i_cat])
    pp.colorbar()
    pp.xticks(np.arange(N),[],fontsize=10)
    pp.yticks(np.arange(N),[],fontsize=10)
    pp.savefig(work_dir+'fin_Q0_pat%d'%i_pat+'.'+grph_fmt,format=grph_fmt)
    pp.close()
    
# matrix plots of (desired) objective covariance patterns
for i_cat in range(n_cat):
    pp.figure(figsize=[2,2])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.imshow(Q0_cat[i_cat,:,:],origin='bottom',interpolation='nearest',vmin=0,vmax=1,cmap=cols[i_cat])
    pp.colorbar(ticks=[0,1])
    pp.xticks(np.arange(N),np.arange(N)+1,fontsize=10)
    pp.yticks(np.arange(N),np.arange(N)+1,fontsize=10)
    pp.savefig(work_dir+'des_Q0_cat%d'%i_cat+'.'+grph_fmt,format=grph_fmt)
    pp.close()


# distributions of output values of variance differences and cross-covariances
v_samp = np.zeros([n_samp,2,2]) # cat; var diff / cross-covar
for i_samp in range(n_samp):
    i_pat = int((i_samp*n_pat)/n_samp)
    i_cat = v_match[i_pat]
    P_sim = noisy_pat(P0_pat[i_pat,:,:])
    Q_sim = np.dot(B_fin,np.dot(P_sim,B_fin.T))
    v_samp[i_samp,i_cat,0] = Q_sim[0,0] - Q_sim[1,1]
    v_samp[i_samp,i_cat,1] = Q_sim[0,1]
    
pp.figure(figsize=[3,3])
pp.axes([0.25,0.6,0.7,0.3])
for i_cat in range(n_cat):
    vp_tmp = pp.violinplot(v_samp[:,i_cat,0],positions=[i_cat],widths=[0.4])
    for lbl_tmp in ('cbars','cmins','cmaxes'):
        vp_tmp[lbl_tmp].set_edgecolor(cols2[i_cat])
    for p_tmp in vp_tmp['bodies']:
        p_tmp.set_facecolor(cols3[i_cat])
        p_tmp.set_edgecolor(cols2[i_cat])
    pp.plot([-1,2],[0]*2,'--k')
pp.xticks([0,1],[],fontsize=10)
pp.ylabel('var diff $Q^0_{00}-Q^0_{11}$',fontsize=10)
pp.yticks([-0.5,0,0.5],fontsize=10)
pp.axis(xmin=-0.4,xmax=1.4,ymin=-0.6,ymax=0.6)
pp.axes([0.25,0.15,0.7,0.3])
for i_cat in range(n_cat):
    vp_tmp = pp.violinplot(v_samp[:,i_cat,1],positions=[i_cat],widths=[0.4])
    for lbl_tmp in ('cbars','cmins','cmaxes'):
        vp_tmp[lbl_tmp].set_edgecolor(cols2[i_cat])
    for p_tmp in vp_tmp['bodies']:
        p_tmp.set_facecolor(cols3[i_cat])
        p_tmp.set_edgecolor(cols2[i_cat])
        p_tmp.set_alpha(0.5)
    pp.plot([-1,2],[0.4]*2,'--k')
pp.xticks([0,1],['cat 1','cat 2'],fontsize=10)
pp.yticks([0,0.5],fontsize=10)
pp.axis(xmin=-0.4,xmax=1.4,ymin=-0.2,ymax=0.8)
pp.ylabel('cross-covar $Q^0_{01}$',fontsize=10)
pp.savefig(work_dir+'discr.'+grph_fmt,format=grph_fmt)
pp.close()




