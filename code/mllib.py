import numpy as np
from scipy import linalg
from utilities import Utilities

import multiprocessing as mp
from time import sleep

#############################################
class MLUtilities(object):
    """ Simple utilities for ML routines. """
    
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
        """ Expect Ypred.shape = Y.shape = (m,n) for n data points and n labels.
            Returns (m,1) vector of scores.
        """
        score = np.sum(Ypred == Y, axis = 1, keepdims = True)
        if score.shape[0] == 1:
            score = score[0,0]
        return score
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

    ###################
    def run_processes(self,tasks,targets,max_procs):
        """ Run at most max_procs concurrent processes. Expect tasks to be list of tuples and targets to be list of
            functions, of equal length.
        """
        if len(tasks) != len(targets):
            raise Exception('Unequal lengths found for tasks ({0:d}) and targets ({1:d})'.format(len(tasks),len(targets)))
        
        # following https://github.com/SaptarshiSrkr/hypersearch/blob/main/hypersearch.py#L139
        active_processes = []
        for r in range(len(tasks)):
            task = tasks[r]
            target = targets[r]
            process = mp.Process(target=target,args=task)
            process.start()

            # Limit concurrent processes
            while len(active_processes) >= max_procs:
                for p in active_processes:
                    if not p.is_alive():
                        active_processes.remove(p)
                sleep(1)

        for p in active_processes:
            p.join()
            
        return
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
