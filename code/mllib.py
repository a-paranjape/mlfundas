import numpy as np
from scipy import linalg
from utilities import Utilities

import multiprocessing as mp
from time import sleep
import gc

import os,psutil
from concurrent.futures import ProcessPoolExecutor

from matplotlib.image import imread

#############################################
class MLUtilities(object):
    """ Simple utilities for ML routines. """
    asc_keys = ['TP','TN','FP','FN','accuracy','precision','recall','F1score']
    asmc_keys = ['confusion','accuracy','precision','recall','F1score']
    
    ###################
    def rv(self,value_list):
        out = np.broadcast_to(value_list,(1,len(value_list)))
        return out.copy()
    ###################
    
    ###################
    def cv(self,value_list):
        row = self.rv(value_list)
        return row.T
    ###################

    ###################
    def one_hot(self,x,k):
        """ One-hot encoding for dataset x of shape (1,n). 
            Expect x to take values in range 1..k, where k >= 1 is integer. 
            Returns array of shape (k,n).
        """
        if len(x.shape) != 2:
            raise ValueError('Expecting x to have shape (1,n) in one_hot().')
        if x.shape[0] != 1:
            raise ValueError('Expecting x to have shape (1,n) in one_hot().')

        n = x.shape[1]
        out = np.zeros((int(k),n),dtype=float)
        for i in range(n):
            out[x[0,i]-1,i] = 1
        return out
    ###################
    
    ###################
    def step_fun(self,z):
        out = np.zeros_like(z)
        out[z > 0.0] = 1.0
        return out
    ###################

    ###################
    def length(self,col_v):
        return np.sqrt(np.sum(col_v**2))
    ###################
    
    ###################
    def normalize(self,col_v):
        len_v = self.length(col_v)
        out = col_v/len_v if len_v > 0.0 else col_v
        return out
    ###################    

    ###################
    def lin_reg(self,x, W, W0):
        """ Expect W0 scalar or (n_l,1), W.shape = (n_{l-1},n_l), x.shape = (n_{l-1},n_{sample}).
            lin_reg(A,W,W0) is pre-activation function for NN layer, with output Z (n_l,n_{sample})
        """
        return W0 + np.dot(W.T,x)
    ###################
    
    ###################
    def positive(self,x, W, W0):
        """ Expect W0 scalar or (1,1), W.shape = (d,1), x.shape = (d,n) for n data points."""
        out = np.sign(self.lin_reg(x,W,W0))
        out[out == 0] = -1.0
        return out
    ###################
    
    ###################
    def score(self,Ypred, Y):
        """ Expect Ypred.shape = Y.shape = (m,n) for n data points and m labels.
            Returns (m,1) vector of scores.
        """
        score = np.sum(Ypred == Y, axis = 1, keepdims = True)
        if score.shape[0] == 1:
            score = score[0,0]
        return score
    ###################

    ###################
    def assess_classification(self,Ypred,Y,neg_labels=True):
        """ Assess binary classification output for predicted labels Ypred and true labels Y.
            Expect Ypred.shape = Y.shape = (1,n) for n data points.
            neg_labels: boolean, True if negative labels used.
            Returns dict with assessment summary having keys:
            TP,TN,FP,FN,accuracy,precision,recall,F1score
        """
        Ypred_i = np.rint(Ypred)
        Y_i = np.rint(Y)
        n_TP = np.where((Ypred_i == 1) & (Y_i == 1))[0].size
        n_TN = np.where((Ypred_i == -neg_labels) & (Y_i == -neg_labels))[0].size
        n_FP = np.where((Ypred_i == 1) & (Y_i == -neg_labels))[0].size
        n_FN = np.where((Ypred_i == -neg_labels) & (Y_i == 1))[0].size
        del Ypred_i,Y_i
        gc.collect()

        accuracy = (n_TP + n_TN)/(n_TP + n_TN + n_FP + n_FN)
        precision = n_TP/(n_TP + n_FP)
        recall    = n_TP/(n_TP + n_FN)
        F1score = 2*precision*recall/(precision + recall + 1e-30)

        out = {'TP':n_TP,'TN':n_TN,'FP':n_FP,'FN':n_FN,
               'accuracy':accuracy,
               'precision':precision,'recall':recall,'F1score':F1score}
        
        if list(out.keys()) != self.asc_keys:
            print('Warning: unexpected list of keys in MLUtilities.assess_classification()')
        
        return out
    ###################

    ###################
    def assess_classification_ensemble(self,neo,X,Y,N_ens_thresh=1000000):
        """ Assess binary classification output for network ensemble. 
            Expect neo to be NetworkEnsembleObject instance, compatible with 
            features X and true labels Y (where X.shape = (nfeat,nsamp) and Y.shape = (1,nsamp)).
            Returns dict with keys
            TP,TN,FP,FN,accuracy,precision,recall,F1score
            containing ensemble (mean,std) or (median,16pc,84pc).
        """
        N_ens = len(neo.keys)
        asc_ens = ({akey:{'mean':0.0,'std':0.0} for akey in self.asc_keys}
                   if N_ens <= N_ens_thresh else
                   {akey:{'median':0.0,'5pc':0.0,'16pc':0.0,'84pc':0.0,'95pc':0.0} for akey in self.asc_keys})
        values = np.zeros((len(self.asc_keys),len(neo.keys)),dtype=float)
        for n in range(len(neo.keys)):
            key = neo.keys[n]
            net_this = neo.ensemble[key]['net']
            asc_this = self.assess_classification(net_this.predict(X),Y,neg_labels=neo.neg_labels)
            for a in range(len(self.asc_keys)):
                akey = self.asc_keys[a]
                values[a,n] = asc_this[akey]
            del net_this

        if N_ens <= N_ens_thresh:
            asc_mean = self.assess_classification(neo.predict(X),Y,neg_labels=neo.neg_labels)
            means = np.mean(values,axis=1)
            stds = np.std(values,axis=1)
            for a in range(len(self.asc_keys)):
                akey = self.asc_keys[a]
                asc_ens[akey]['mean'] = asc_mean[akey] # means[a]
                asc_ens[akey]['std'] = stds[a]
        else:
            medians = np.median(values,axis=1)
            pcs_05 = np.percentile(values,5,axis=1)
            pcs_16 = np.percentile(values,16,axis=1)
            pcs_84 = np.percentile(values,84,axis=1)
            pcs_95 = np.percentile(values,95,axis=1)

            for a in range(len(self.asc_keys)):
                akey = self.asc_keys[a]
                asc_ens[akey]['median'] = medians[a]
                asc_ens[akey]['5pc'] = pcs_05[a]
                asc_ens[akey]['16pc'] = pcs_16[a]
                asc_ens[akey]['84pc'] = pcs_84[a]
                asc_ens[akey]['95pc'] = pcs_95[a]
                
        return asc_ens
    ###################


    ###################
    def assess_multi_classification(self,Ypred,Y):
        """ Assess binary classification output for predicted labels Ypred and true labels Y.
            Expect Ypred.shape = Y.shape = (K,n) for K categories and n data points, domain {0,1} (one-hot encoding).
            Returns dict with assessment summary having keys:
            confusion,accuracy,precision,recall,F1score
            where confusion = confusion matrix and remaining quantities are equi-weighted macro-averages.
        """
        Ypred_i = np.rint(Ypred)
        Y_i = np.rint(Y)
        
        K = Y.shape[0]
        confusion = np.zeros((K,K),dtype=float)
        # rows: true values, columns: predicted values
        for k_true in range(K):
            for k_pred in range(K):
                count = np.where((Y_i[k_true] == 1.0) & (Ypred_i[k_pred] == 1.0))[0].size
                confusion[k_true,k_pred] = count
        
        del Ypred_i,Y_i
        gc.collect()

        if np.sum(confusion) != Y.shape[1]:
            raise Exception("Invalid sum over confusion matrix elements.")

        accuracy = np.trace(confusion)/Y.shape[1]

        precision_k = np.zeros(K)
        recall_k = np.zeros(K)
        for k in range(K):
            precision_k[k] = confusion[k,k]/(np.sum(confusion[:,k]) + 1e-30)
            recall_k[k] = confusion[k,k]/(np.sum(confusion[k,:]) + 1e-30)
        
        precision = np.mean(precision_k)
        recall    = np.mean(recall_k)
        F1score = 2*precision*recall/(precision + recall + 1e-30)

        out = {'confusion':confusion,'accuracy':accuracy,'precision':precision,'recall':recall,'F1score':F1score}
        
        if list(out.keys()) != self.asmc_keys:
            print('Warning: unexpected list of keys in MLUtilities.assess_multi_classification()')
        
        return out
    ###################


    ###################
    def assess_multi_classification_ensemble(self,neo,X,Y,N_ens_thresh=1000000):
        """ Assess multi-class classification output for network ensemble. 
            Expect neo to be NetworkEnsembleObject instance, compatible with 
            features X and true labels Y (where X.shape = (nfeat,nsamp) and Y.shape = (K,nsamp)).
            Returns dict with keys
            confusion,accuracy,precision,recall,F1score
            containing ensemble (mean,std) or (median,16pc,84pc).
        """
        N_ens = len(neo.keys)
        asmc_ens = ({akey:{'mean':0.0,'std':0.0} for akey in self.asmc_keys}
                   if N_ens <= N_ens_thresh else
                   {akey:{'median':0.0,'5pc':0.0,'16pc':0.0,'84pc':0.0,'95pc':0.0} for akey in self.asmc_keys})
        confusions = np.zeros((len(neo.keys),Y.shape[0],Y.shape[0]),dtype=float)
        values = np.zeros((len(self.asmc_keys)-1,len(neo.keys)),dtype=float)
        for n in range(len(neo.keys)):
            key = neo.keys[n]
            net_this = neo.ensemble[key]['net']
            asmc_this = self.assess_multi_classification(net_this.predict(X),Y)
            confusions[n] = asmc_this['confusion']
            for a in range(len(self.asmc_keys)-1):
                akey = self.asmc_keys[a+1]
                values[a,n] = asmc_this[akey]
            del net_this

        if N_ens <= N_ens_thresh:
            asmc_mean = self.assess_multi_classification(neo.predict(X),Y)
            asmc_ens['confusion']['mean'] = asmc_mean['confusion'] #np.mean(confusions,axis=0)
            asmc_ens['confusion']['std'] = np.std(confusions,axis=0)
            means = np.mean(values,axis=1)
            stds = np.std(values,axis=1)
            for a in range(len(self.asmc_keys)-1):
                akey = self.asmc_keys[a+1]
                asmc_ens[akey]['mean'] = asmc_mean[akey] # means[a]
                asmc_ens[akey]['std'] = stds[a]
        else:
            asmc_ens['confusion']['median'] = np.median(confusions,axis=0)
            asmc_ens['confusion']['5pc'] = np.percentile(confusions,5,axis=0)
            asmc_ens['confusion']['16pc'] = np.percentile(confusions,16,axis=0)
            asmc_ens['confusion']['84pc'] = np.percentile(confusions,84,axis=0)
            asmc_ens['confusion']['95pc'] = np.percentile(confusions,95,axis=0)

            medians = np.median(values,axis=1)
            pcs_05 = np.percentile(values,5,axis=1)
            pcs_16 = np.percentile(values,16,axis=1)
            pcs_84 = np.percentile(values,84,axis=1)
            pcs_95 = np.percentile(values,95,axis=1)
            
            for a in range(len(self.asmc_keys)-1):
                akey = self.asmc_keys[a+1]
                asmc_ens[akey]['median'] = medians[a]
                asmc_ens[akey]['5pc'] = pcs_05[a]
                asmc_ens[akey]['16pc'] = pcs_16[a]
                asmc_ens[akey]['84pc'] = pcs_84[a]
                asmc_ens[akey]['95pc'] = pcs_95[a]
                    
        del values,confusions
        
        return asmc_ens
    ###################

    ###################
    def prime_factors(self,n):
        """Returns all the prime factors of a positive integer. 
           Taken from https://stackoverflow.com/questions/23287/algorithm-to-find-largest-prime-factor-of-a-number/412942#412942
        """
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n = n//d
            d = d + 1
            if d*d > n:
                if n > 1: factors.append(n)
                break
        return np.array(factors)
    ###################
    
    
    ###################
    # helper functions to re-package images into vectors
    ###################
    def unroll(self,I):
        """ Single binary clockwise unroll. Expect I to be numpy array of shape (2n,2n) for integer n. """
        n = I.shape[0]//2
        I0 = I[n:,:n]
        I1 = I[n:,n:]
        I2 = I[:n,n:]
        I3 = I[:n,:n]
        return [I0,I1,I2,I3]
    ###################
    
    ###################
    def vortex_unroll(self,in_list,first_use=True):
        """ Binary recursive clockwise unroll. Expect in_list to be list of numpy arrays of shape (D,D) for integer D. 
            If D is odd, reverts to standard numpy flatten, else performs vortex unrolling.
        """
        if len(in_list) == 0:
            raise Exception('Need non-empty list in vortex_unroll()')
        
        out_list = []
        D = in_list[0].shape[0]
        
        if first_use:
            for I in in_list:
                if I.shape != (D,D):
                    err_str = 'Incompatible matrix detected in binary_recursive_unroll(). '
                    err_str += "Expecting ({0:d},{0:d}), found ".format(D)
                    err_str += '(' + ','.join([str(i) for i in I.shape])  + ')'
                    raise Exception(err_str)
                
        if (D % 2) != 0:
            for I in in_list:
                out_list.append(I.flatten())
            out_list = np.array(out_list).flatten()
        else:
            pfs = self.prime_factors(D)
            m = np.where(pfs == 2)[0].size # guaranteed m > 0
            for I in in_list:
                Js = self.unroll(I)
                for J in Js:
                    out_list.append(J)
            out_list = self.vortex_unroll(out_list,first_use=False)
            
        return np.squeeze(np.array(out_list))
    ###################

    
    ###################
    def vortex_package(self,X):
        """ Re-package array of matrices X of shape (D,D,nsamp)
            into array of shape (D**2,nsamp), where each D x D matrix is unrolled using 
            vortex_unroll which partially preserves spatial coherence.
        """
        if len(X.shape) != 3:
            raise Exception("vortex_package() needs input of shape (D,D,nsamp).")
        if X.shape[0] != X.shape[1]:
            raise Exception("vortex_package() needs input of shape (D,D,nsamp).")
        D = X.shape[0]

        nsamp = X.shape[2]
        X_out = np.zeros((nsamp,D**2),dtype=X.dtype) # will be transposed later

        for i in range(nsamp):
            X_out[i] = self.vortex_unroll([X[:,:,i]])

        X_out = X_out.T # shape (D**2,nsamp)
        return X_out
    ###################
    ###################

    
    ###################
    def tanh(self,x):
        return np.tanh(x)

    # Where th is the output
    def tanh_gradient(self,th):
        return 1 - th**2
    ###################

    ###################
    def clip(self,x,c):
        """ Clip x to the range [-c,c]. Expect x to be array-like and c to be positive scalar. 
            Returns array of x.shape truncated to this range, with outliers replaced by boundary values.
        """
        c_ones = c*np.ones_like(x)
        return np.minimum(np.maximum(x,-1.0*c_ones),c_ones)
    ###################

    ###################
    def FID(self,X1,X2):
        """ Calculate Frechet Inception Distance (FID) between two data sets.
            Based on eqn 6 of Heusel+ arXiv:1706.08500.
            -- X1, X2: data sets of shape (d,n1), (d,n2) respectively.
            Returns scalar value of FID.
        """
        nd = X1.shape[0]
        if X2.shape[0] != nd:
            raise Exception("Expecting equal first dimension of X1 and X2 in FID(). Got d1 = {0:d}, d2 = {1:d}".format(X1.shape[0],X2.shape[0]))
        
        mu1 = np.mean(X1,axis=1)
        cov1 = np.cov(X1)
        mu2 = np.mean(X2,axis=1)
        cov2 = np.cov(X2)

        fid = np.sum((mu1 - mu2)**2)
        fid += (np.trace(cov1 + cov2 - 2*linalg.sqrtm(np.dot(cov1,cov2))) if nd > 1 else (cov1 + cov2 - 2*np.sqrt(cov1*cov2)))

        return fid
    ###################


    ###################
    def KLGauss(self,X1,X2):
        """ Calculate Kullbach-Liebler divergence D(X1||X2) between two data sets, assuming each is Gaussian distributed.
            Currently assumes cov(X2) is invertible.
            -- X1, X2: data sets of shape (d,n1), (d,n2) respectively.
            Returns scalar value of KL divergence.
        """
        nd = X1.shape[0]
        if X2.shape[0] != nd:
            raise Exception("Expecting equal first dimension of X1 and X2 in KLGauss(). Got d1 = {0:d}, d2 = {1:d}"
                            .format(X1.shape[0],X2.shape[0]))
        
        mu1 = np.mean(X1,axis=1)
        cov1 = np.cov(X1)
        mu2 = np.mean(X2,axis=1)
        cov2 = np.cov(X2)

        kld = self.KLGauss_Dist(mu1,mu2,cov1,cov2)
        
        return kld
    ###################

    ###################
    def KLGauss_Dist(self,mu1,mu2,cov1,cov2):
        """ Calculate Kullbach-Liebler divergence D(N1||N2) between two Gaussian distributions
            with (mean,cov) given by (mu1,cov1) and (mu2,cov2), respectively.
            Currently assumes cov2 is invertible.
            -- mu1, mu2: means of shape (d,).
            -- cov1,cov2: covariance matrices of shape (d,d)
            Returns scalar value of KL divergence.
        """
        nd = mu1.size
        if mu1.shape != (nd,):
            raise Exception('Invalid shape for mu1 in KLGauss_Dist().')
        if mu2.shape != (nd,):
            raise Exception('Invalid shape for mu2 in KLGauss_Dist().')
        if cov1.shape != (nd,nd):
            raise Exception('Invalid shape for cov1 in KLGauss_Dist().')
        if cov2.shape != (nd,nd):
            raise Exception('Invalid shape for cov2 in KLGauss_Dist().')
        
        if nd > 1: 
            inv2 = linalg.inv(cov2)  
            kld = np.dot(mu1-mu2,np.dot(inv2,mu1-mu2))
            kld += (np.trace(np.dot(cov1,inv2)) - nd - np.log(linalg.det(cov1)/(linalg.det(cov2)+1e-15)))
        else:
            kld = (mu1-mu2)**2/(cov2 + 1e-15)
            kld += cov1/(cov2 + 1e-15) - 1 - np.log(cov1/(cov2 + 1e-15))

        kld *= 0.5

        return kld
    ###################
    

    ###################
    def nongauss_diff(self,X1,X2,mom=3):
        """ Calculate mean difference in dimension-wise non-Gaussian moments between 2 data sets.
            -- X1, X2: data sets of shape (d,n1), (d,n2) respectively.
            Returns scalar value of mean of absolute difference in non-Gaussian moment along each direction.
        """
        nd = X1.shape[0]
        if X2.shape[0] != nd:
            raise Exception("Expecting equal first dimension of X1 and X2 in skew_diff(). Got d1 = {0:d}, d2 = {1:d}"
                            .format(X1.shape[0],X2.shape[0]))
        
        mu1 = np.mean(X1,axis=1)
        ng1 = np.mean((X1 - mu1)**mom,axis=1)/(np.std(X1,axis=1)**mom + 1e-15)
        
        mu2 = np.mean(X2,axis=1)
        ng2 = np.mean((X2 - mu2)**mom,axis=1)/(np.std(X2,axis=1)**mom + 1e-15)

        mom_diff = np.mean(np.fabs(ng1 - ng2))
        return mom_diff
    ###################

    
    # ###################
    # # Example queuer
    # ###################
    # def queuer(self,r,X,Y,params,net,mdict):
    #     """ Wrapper to execute Sequential.train on copy of Sequential instance and add the instance into a managed dictionary. """
    #     net.train(X,Y,params)
    #     mdict[r+1] = net
    # ###################    
    

    ###################
    def run_processes_alt(self,tasks,queuer,max_procs):
        """ General purpose routine to run at most max_procs concurrent processes. 
            -- tasks: list of tuples of form 1. (arg1,..argn,method/instance)
                          Case 1.: method is (un)bound function instance with call signature method(arg1,..,argn)
                          Case 2.: instance is a class instance and queuer should internally define 
                                   method = getattr(class_instance,some_method_name) having call signature method(arg1,..,argn)
                          E.g.: in case 2 we might set tasks = (X,Y,params,net), where 
                                net is a Sequential instance and X,Y,params are inputs to net.train(). 
            -- queuer: target function with call signature (r,arg1,..,argn,method/instance,mdict)
                       where r is the integer index of a task and mdict is a common managed dictionary, i.e. instance of multiprocessing.Manager.dict() 
                       that can be used internally by the queuer to pass arbitrary data structures from a child to the parent process.
                       If instance passed instead of method, queuer should internally use appropriate method of the instance.
            -- max_procs: int, maximum number of concurrent processes.

            Returns updated mdict.
        """
        manager = mp.Manager()
        mdict = manager.dict({r+1:None for r in range(len(tasks))}) # need this to pass around class instances
        
        # futures = []
        a = 0
        ###################################
        # context 'spawn' would access multiple processors. default is 'fork' which doesn't.
        # can't use spawn however since ppe.submit apparently only accepts imported functions, not class instance methods or even functions defined within the script.
        ###################################
        # with ProcessPoolExecutor(max_workers=max_procs,max_tasks_per_child=None,mp_context=mp.get_context("spawn")) as ppe:
        with ProcessPoolExecutor(max_workers=max_procs,max_tasks_per_child=None) as ppe:
            for r in range(len(tasks)):
                args = (r,)+tasks[r]+(mdict,)
                future = ppe.submit(queuer,*args)

                a += 1
                if a == max_procs:
                    a -= max_procs
            
        return mdict
    ###################

    
    ###################
    def run_processes(self,tasks,queuer,max_procs):
        """ General purpose routine to run at most max_procs concurrent processes. 
            -- tasks: list of tuples of form 1. (arg1,..argn,method/instance)
                          Case 1.: method is (un)bound function instance with call signature method(arg1,..,argn)
                          Case 2.: instance is a class instance and queuer should internally define 
                                   method = getattr(class_instance,some_method_name) having call signature method(arg1,..,argn)
                          E.g.: in case 2 we might set tasks = (X,Y,params,net), where 
                                net is a Sequential instance and X,Y,params are inputs to net.train(). 
            -- queuer: target function with call signature (r,arg1,..,argn,method/instance,mdict)
                       where r is the integer index of a task and mdict is a common managed dictionary, 
                       i.e. instance of multiprocessing.Manager.dict() 
                       that can be used internally by the queuer to pass arbitrary data structures from a child to the parent process.
                       If instance passed instead of method, queuer should internally use appropriate method of the instance.
            -- max_procs: int, maximum number of concurrent processes.

            Returns updated mdict.
        """
        default_method = mp.get_start_method()
        mp.set_start_method("spawn",force=True)
        
        manager = mp.Manager()
        mdict = manager.dict({r+1:None for r in range(len(tasks))}) # need this to pass around class instances
        
        # loop structure inspired by https://github.com/SaptarshiSrkr/hypersearch/blob/main/hypersearch.py#L139
        active_processes = []
        a = 0
        for r in range(len(tasks)):
            process = mp.Process(target=queuer,args=(r,)+tasks[r]+(mdict,))
            process.start()
            active_processes.append(process)
            
            # Limit concurrent processes
            while len(active_processes) >= max_procs:
                for p in active_processes:
                    if not p.is_alive():
                        active_processes.remove(p)
                sleep(1)

            a += 1
            if a == max_procs:
                a -= max_procs

        for p in active_processes:
            p.join()
            
        mp.set_start_method(default_method,force=True)
        return mdict
    ###################
    

    ###################
    def run_processes_batch(self,tasks,queuer,max_procs):
        """ General purpose routine to run at most max_procs concurrent processes. (ALTERNATE SCHEDULING.)
            -- tasks: list of tuples of form 1. (arg1,..argn,method/instance)
                          Case 1.: method is (un)bound function instance with call signature method(arg1,..,argn)
                          Case 2.: instance is a class instance and queuer should internally define 
                                   method = getattr(class_instance,some_method_name) having call signature method(arg1,..,argn)
                          E.g.: in case 2 we might set tasks = (X,Y,params,net), where 
                                net is a Sequential instance and X,Y,params are inputs to net.train(). 
            -- queuer: target function with call signature (r,arg1,..,argn,method/instance,mdict)
                       where r is the integer index of a task and mdict is a common managed dictionary, i.e. instance of multiprocessing.Manager.dict() 
                       that can be used internally by the queuer to pass arbitrary data structures from a child to the parent process.
                       If instance passed instead of method, queuer should internally use appropriate method of the instance.
            -- max_procs: int, maximum number of concurrent processes.

            Returns updated mdict.
        """
        manager = mp.Manager()
        mdict = manager.dict({r+1:None for r in range(len(tasks))}) # need this to pass around class instances


        ntasks = len(tasks)
        if ntasks > max_procs:
            n_concur = 1*max_procs
            n_batches = ntasks // n_concur
            n_tasks_last_batch = ntasks % n_concur
            if n_tasks_last_batch > 0:
                n_batches += 1
        else:
            n_concur = 1*ntasks
            n_batches = 1
            n_tasks_last_batch = n_concur

        r = 0
        for b in range(n_batches):
            n_task_this_batch = n_tasks_last_batch if (b == n_batches-1) else n_concur
            if n_task_this_batch > 0:
                processes = []
                for n in range(n_task_this_batch):
                    task = tasks[r]
                    process = mp.Process(target=queuer,args=(r,)+task+(mdict,))
                    process.start()
                    processes.append(process)
                    r += 1
                    
                for p in processes:
                    p.join()
                
        return mdict
    ###################

    #################################
    # MNIST manipulation code courtesy MIT-OLL MLIntro Course
    ###################
    def load_mnist_data(self,labels,path):
        """
        @param labels list of labels from {0, 1,...,9}
        @param path str path to data/mnist folder
        @return dict: label (int) -> [[image1], [image2], ...]
        """

        data = {}

        for label in labels:
            images = self.load_mnist_single(path+"/data/mnist/mnist_train{}.png".format(label))
            y = np.array([[label] * len(images)])
            data[label] = {
                "images": images,
                "labels": y
            }

        return data
    ###################

    ###################
    def load_mnist_single(self,path_data):
        """
        @return list of images (first row of large picture)
        """

        img = imread(path_data)  # 2156 x 2156 (m,n)
        m, n = img.shape

        side_len = 28  # standard mnist
        n_img = int(m / 28)

        imgs = []  # list of single images
        for i in range(n_img):
            start_ind = i*side_len
            end_ind = start_ind + side_len
            current_img = img[start_ind:end_ind, :side_len]  # 28 by 28

            current_img = current_img / 255 # normalization!!!

            imgs.append(current_img)

        return imgs
    ###################


    
#################################
    
    
#################################
# (structure courtesy MIT-OLL MLIntro Course)
# Discrete distribution represented as a dictionary.  Can be
# sparse, in the sense that elements that are not explicitly
# contained in the dictionary are assumed to have zero probability.
#################
class DDist(object):
    """ Discrete distribution over states. """
    def __init__(self, dictionary,rng=None):
        # Initializes dictionary whose keys are elements of the domain
        # and values are their probabilities
        self.ddist = dictionary
        self.support = list(self.ddist.keys())
        self.rng = rng if rng is not None else np.random.RandomState()

    def prob(self, elt):
        # Returns the probability associated with elt
        return self.ddist[elt]

    def support(self):
        # Returns a list (in any order) of the elements of this
        # distribution with non-zero probability.
        return self.support

    def draw(self):
        # Returns a randomly drawn element from the distribution
        u = self.rng.rand()
        for elt in self.support:
            prob = self.ddist[elt]
            if u < prob:
                return elt
            else:
                u -= prob
        raise Exception('Failed to draw from '+ str(self))

    def expectation(self, f):
        # Returns the expected value of the function f over the current distribution
        return sum([self.ddist[elt]*f(elt) for elt in self.support])
#################################

#############################################
class SeqUtilities(object):
    
    def uniform_dist(self,elts):
        """
        Uniform distribution over a given finite set of C{elts}
        @param elts: list of any kind of item
        """
        p = 1.0 / len(elts)
        return DDist(dict([(e, p) for e in elts]))
    
    def value(self,q,s):
        val = 0.0
        for a in q.actions:
            val = np.max([val,q.get(s,a)])
        return val

    def greedy(self,q,s):
        val = 0.0
        act = None
        for a in q.actions:
            val_this = q.get(s,a)
            if val_this > val:
                val = val_this
                act = a
        return act if act is not None else q.actions[0]

    def epsilon_greedy(self,q,s,eps=0.5,rng=None):
        u = rng.rand() if rng is not None else np.random.rand()
        if u < eps:
            return self.uniform_dist(q.actions).draw()
        else:
            return self.greedy(q,s)
#################################

    
#############################################
class Evaluate(MLUtilities,Utilities):
    def __init__(self,verbose=True,logfile=None):
        self.verbose = verbose
        self.logfile = logfile

    def eval_classifier(self,learner,X_train,Y_train,X_test,Y_test,params={}):
        learner.train(X_train,Y_train,params=params)
        Y_pred = learner.predict(X_test)
        perc = self.score(Y_pred,Y_test)
        perc /= X_test.shape[1]
        return perc
    
    def eval_learning_alg(self,learner,data_gen,n_train,n_test,it,gen_test=False,params={}):        
        if self.verbose:
            self.print_this('Evaluating algorithm...',self.logfile)
        perc = np.zeros(it)
        for i in range(it):
            X_train,Y_train = data_gen(n_train)
            X_test,Y_test = data_gen(n_test) if gen_test else (X_train,Y_train)
            perc[i] = self.eval_classifier(learner,X_train,Y_train,X_test,Y_test,params=params)
            if self.verbose:
                self.status_bar(i,it)
        if self.verbose:
            self.print_this('... done',self.logfile)
        
        return perc.mean(),perc.std()/np.sqrt(it-1 + 1e-8)

    def xval_learning_alg(self,learner,X,Y,k,params={}):
        X_split = np.array_split(X,k,axis=1)
        Y_split = np.array_split(Y,k,axis=1)
        perc = np.zeros(k)
        for j in range(k):
            X_minus_j = np.concatenate(X_split[:j]+X_split[j+1:],axis=1)
            Y_minus_j = np.concatenate(Y_split[:j]+Y_split[j+1:],axis=1)
            perc[j] = self.eval_classifier(learner,X_minus_j,Y_minus_j,X_split[j],Y_split[j],params=params)
            if self.verbose:
                self.status_bar(j,k)
        if self.verbose:
            self.print_this('... done',self.logfile)
        return perc.mean(),perc.std()/np.sqrt(k-1 + 1e-8)
    
    ##################################################
    # courtesy MIT-OLL IntroML Course
    def gen_lin_separable(self,num_points=20, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
        ''' 
        Generate linearly separable dataset X, y given theta and theta0
        Return X, y where
        X is a numpy array where each column represents a dim-dimensional data point
        y is a column vector of 1s and -1s
        '''
        X = np.random.uniform(low=-5, high=5, size=(dim, num_points))
        y = np.sign(np.dot(np.transpose(th), X) + th_0)
        return X, y

    def gen_flipped_lin_separable(self,num_points=20, pflip=0.25, 
                                  th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
        '''
        Generate difficult (usually not linearly separable) data sets by
        "flipping" labels with some probability.
        Returns a method which takes num_points and flips labels with pflip
        '''
        def flip_generator(num_points=num_points):
            X, y = self.gen_lin_separable(num_points, th, th_0, dim)
            flip = np.random.uniform(low=0, high=1, size=(num_points,))
            for i in range(num_points):
                if flip[i] < pflip: y[0,i] = -y[0,i]
            return X, y
        return flip_generator
#############################################
