import time
import numpy as np
import matplotlib.pyplot as pp

#%% init

work_dir = 'mnist5digits_covperc/'

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

# number of input patterns for training
v_n_pat = np.array([500,5000,50000],dtype=np.int)
n_n_pat = v_n_pat.size


#%% network model

M = 2 * nrows
N = n_cat

n_shift = 1
T = nrows + n_shift # move all digit till end

mask_offdiag = np.logical_not(np.eye(N,dtype=bool))
mask_tri = np.tri(N,N,-1,dtype=np.bool)


# compute network activity
def sim_net(B_arg,I_arg):
    x_tmp = I_arg
    y_tmp = np.dot(B_arg,x_tmp)
    return x_tmp, y_tmp

# compute input/output covariances from time series
def comp_cov_emp(ts_x_arg,ts_y_arg,T_arg):
    P0_tmp = np.tensordot(ts_x_arg[:,0:T_arg-1],ts_x_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-2)
    Q0_tmp = np.tensordot(ts_y_arg[:,0:T_arg-1],ts_y_arg[:,0:T_arg-1],axes=(1,1)) / (T_arg-2)
    return P0_tmp, Q0_tmp

# simulate left/right moving in inputs (receptor field)
def get_input(data_arg, i_pat_arg, n_arg):
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

# basis matrices; 1 for indices (i,k), 0 otherwise
M_ik = np.zeros([N,M,N,M])
for i in range(N):
    for k in range(M):
        M_ik[i,k,i,k] = 1

# test accuracy
def test_acc(B_arg, data_arg, labels_arg, subset_ind_arg, n_pat_arg):
    acc_tmp = 0
    CM_tmp = np.zeros([n_cat,n_cat])
    for i_pat in np.arange(n_pat_arg)[subset_ind_arg]:
        # input and label
        I = get_input(data_arg, i_pat, n_pat_arg)
        i_cat = labels_arg[i_pat]
        # simulate activity and calculate empirical covriances
        x_sim,y_sim = sim_net(B_arg,I)
        P0_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,T)
        # test on output variance
        acc_tmp += int(i_cat==np.argmax(Q0_sim.diagonal()))
        CM_tmp[i_cat,np.argmax(Q0_sim.diagonal())] += 1
    # returns accuracy and confusion matrix
    return acc_tmp / subset_ind_arg.sum(), CM_tmp


#%% optimization

n_opt = 20

n_rep = 10 # repeating same experiment
    

# save error history and accuracy
B_hist = np.zeros([n_n_pat,n_opt+1,N,M])
acc_summary = np.zeros([n_n_pat,n_rep,n_opt+1,2])
CM = np.zeros([n_n_pat,n_rep,n_opt+1,2,n_cat,n_cat])

t_start = time.time()

# loop over all train patterns
for i_n_pat in range(n_n_pat):
    n_pat_train = v_n_pat[i_n_pat]
    n_pat_test = int(n_pat_train/10)
    
    # rescaling learning rate
    eta_B = 10 / n_pat_train

    # repeat similar configuration
    for i_rep in range(n_rep):

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
        
        
        # initial conditions
        B = np.random.randn(N,M) * 0.3
        if i_rep==0:
            B_hist[i_n_pat,0,:,:] = B
        
        # perf on train patterns with initial conditions
        acc_tmp, CM_tmp = test_acc(B, train_data, train_label, subset_train, n_train)
        acc_summary[i_n_pat,i_rep,0,0] = acc_tmp
        CM[i_n_pat,i_rep,0,0,:,:] = CM_tmp

        # perf on test patterns with initial conditions
        acc_tmp, CM_tmp = test_acc(B, test_data, test_label, subset_test, n_test)
        acc_summary[i_n_pat,i_rep,0,1] = acc_tmp
        CM[i_n_pat,i_rep,0,1,:,:] = CM_tmp

        print('train err, test err:', acc_summary[i_n_pat,i_rep,0,0], acc_summary[i_n_pat,i_rep,0,1])
            
        # loop over optimization epochs
        for i_opt in range(n_opt):
            print('n_pat, rep, opt:', i_n_pat, i_rep, i_opt)
        
            # loop over all training patterns
            for i_pat in np.random.permutation(np.arange(n_train)[subset_train]):
                
                # input and label
                I = get_input(train_data, i_pat, n_train)
                i_cat = train_label[i_pat]
                Q0_cat = np.eye(N) * 0.2
                Q0_cat[i_cat,i_cat] = 1
                
                # simulate activity and calculate empirical covriances
                x_sim,y_sim = sim_net(B,I)
                P0_sim, Q0_sim = comp_cov_emp(x_sim,y_sim,T)
                # error on output cov
                delta_Q0_sim = Q0_cat - Q0_sim
                # with output masking
                delta_Q0_sim[mask_offdiag] = 0
                # deriv Q0 wrt B
                d_Q0_B = np.einsum('imkl, jm -> ijkl', M_ik, np.dot(B,P0_sim)) + np.einsum('jmkl, im -> ijkl', M_ik, np.dot(B,P0_sim))
                # weight update
                B += eta_B * np.einsum('jl, jlik -> ik', delta_Q0_sim, d_Q0_B)
                                                        
            # store weights
            if i_rep==0:
                B_hist[i_n_pat,i_opt+1,:,:] = B

            # perf on train patterns
            acc_tmp, CM_tmp = test_acc(B, train_data, train_label, subset_train, n_train)
            acc_summary[i_n_pat,i_rep,i_opt+1,0] = acc_tmp
            CM[i_n_pat,i_rep,i_opt+1,0,:,:] = CM_tmp
    
            # perf on test patterns with initial conditions
            acc_tmp, CM_tmp = test_acc(B, test_data, test_label, subset_test, n_test)
            acc_summary[i_n_pat,i_rep,i_opt+1,1] = acc_tmp
            CM[i_n_pat,i_rep,i_opt+1,1,:,:] = CM_tmp
                                    
            print('train err, test err:', acc_summary[i_n_pat,i_rep,i_opt+1,0], acc_summary[i_n_pat,i_rep,i_opt+1,1])
            min_tmp = int((time.time() - t_start)/60)
            sec_tmp = int(time.time() - t_start) - 60 * min_tmp
            print('time since start:', min_tmp, 'min,', sec_tmp, 'sec')
                
np.save(work_dir+'acc_summary.npy',acc_summary)
np.save(work_dir+'CM.npy',CM)


#%% plots
if False: # load saved results
    acc_summary = np.load(work_dir+'acc_summary.npy')
    CM = np.load(work_dir+'CM.npy')
    
label_aff = np.concatenate((np.arange(n_digits),np.arange(n_digits)))
label_aff = np.array(label_aff, dtype='str')
for i in range(n_digits):
    label_aff[i] += 'r'
    label_aff[i+n_digits] += 'l'
    

pp.figure(figsize=[3,2])
pp.axes([0.2,0.25,0.7,0.7])
for i_n_pat in range(n_n_pat):
    acc_tmp = acc_summary[i_n_pat,:,:,0] # train
    pp.fill_between(range(n_opt+1),acc_tmp.mean(0)-acc_tmp.std(0)/np.sqrt(n_rep),acc_tmp.mean(0)+acc_tmp.std(0)/np.sqrt(n_rep),color=[0.8,0.8,0.8])
    pp.plot(range(n_opt+1),acc_tmp.mean(0),ls=':',lw=1.5,color=cols_gr[i_n_pat]) # mean
    acc_tmp = acc_summary[i_n_pat,:,:,1] # test
    pp.fill_between(range(n_opt+1),acc_tmp.mean(0)-acc_tmp.std(0)/np.sqrt(n_rep),acc_tmp.mean(0)+acc_tmp.std(0)/np.sqrt(n_rep),color=[0.8,0.8,0.8])
    pp.plot(range(n_opt+1),acc_tmp.mean(0),ls='-',lw=1.5,color=cols_gr[i_n_pat]) # mean
pp.legend(['train','test'],fontsize=7)
pp.plot([-1,n_opt+1],[1/n_cat]*2,'--k')
pp.xticks(fontsize=10)
pp.yticks([0,0.5,1],fontsize=10)
pp.axis(xmin=-0.4,xmax=n_opt+0.4,ymin=0,ymax=1)
pp.xlabel('optimization epoch',fontsize=10)
pp.ylabel('accuracy',fontsize=10)
pp.savefig(work_dir+'acc_train_test.'+grph_fmt,format=grph_fmt)
pp.close()


# examples for one repetition
i_rep = 0
for i_n_pat in range(n_n_pat):
    for i_opt in np.arange(0,n_opt+1,2):
        pp.figure(figsize=[2,2])
        pp.axes([0.3,0.3,0.65,0.6])
        pp.imshow(CM[i_n_pat,i_rep,i_opt,0,:,:], origin='bottom', cmap='Greys')
        pp.xticks([0,4,5,9],label_aff[[0,4,5,9]],fontsize=10)
        pp.yticks([0,4,5,9],label_aff[[0,4,5,9]],fontsize=10)
        pp.xlabel('predicted label',fontsize=10)
        pp.ylabel('true label',fontsize=10)
        pp.savefig(work_dir+'CM_train_n'+str(i_n_pat)+'_opt'+str(i_opt)+'.'+grph_fmt,format=grph_fmt)
        pp.close()
    
        pp.figure(figsize=[2,2])
        pp.axes([0.3,0.3,0.65,0.6])
        pp.imshow(CM[i_n_pat,i_rep,i_opt,1,:,:], origin='bottom', cmap='Greys')
        pp.xticks([0,4,5,9],label_aff[[0,4,5,9]],fontsize=10)
        pp.yticks([0,4,5,9],label_aff[[0,4,5,9]],fontsize=10)
        pp.xlabel('predicted label',fontsize=10)
        pp.ylabel('true label',fontsize=10)
        pp.savefig(work_dir+'CM_test_n'+str(i_n_pat)+'_opt'+str(i_opt)+'.'+grph_fmt,format=grph_fmt)
        pp.close()

# examples of weight evolution with and without output masking
for i_n_pat in range(n_n_pat):
    pp.figure(figsize=[2,2])
    pp.axes([0.3,0.3,0.65,0.6])
    pp.plot(np.arange(n_opt+1),B_hist[i_n_pat,:,:,:].reshape([n_opt+1,N*M]))
    pp.xticks(fontsize=10)
    pp.yticks(fontsize=10)
    pp.xlabel('optimization epoch',fontsize=10)
    pp.ylabel('aff weight',fontsize=10)
    pp.savefig(work_dir+'B_hist_n'+str(i_n_pat)+'.'+grph_fmt,format=grph_fmt)
    pp.close()
    