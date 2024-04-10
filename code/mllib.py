import numpy as np
from utilities import Utilities
import copy

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

#############################################
class Perceptron(MLUtilities,Utilities):
    def __init__(self,origin=False,init=None,avg=False,verbose=True,hook=None,logfile=None):
        """ Perceptron algorithm.
            origin: whether to classify through origin (True) or include offset (False, default)
            init: None 
                  or initial th [shape (d,1)] (if origin==True) 
                  or tuple with initial (th [shape (d,1)],th0 [shape (1,1)]).
            avg: whether to use standard or averaged perceptron (Default False, standard perceptron)
        """
        self.origin = origin
        self.verbose = verbose
        self.logfile = logfile
        self.hook = hook
        self.init = init
        self.avg = avg

        if self.avg & self.origin:
            self.print_this("Averaged perceptron only implemented with offset. Setting origin = False.")
            self.origin = False

        self.train = self.avg_perceptron if self.avg else self.perceptron 

        
    def perceptron(self,X,Y,params={}):
        """ Perceptron algorithm.
            X: (d,n) array for n data points in d dimensions
            Y: (1,n) array containing +-1.
            params: hyperparameter dictionary.
            origin: whether to classify through origin (True) or include offset (False, default)
        """
        T = params.get('T',10000000)
        self.epochs = np.arange(T)+1.0
        self.epoch_loss = np.zeros_like(self.epochs)
        
        d,n = X.shape
        if Y.shape != (1,n):
            raise TypeError('incompatible features and labels in perceptron(). Need X = (d,n), Y = (1,n).')
            
        if self.init is None:
            self.W = self.cv([0.0]*d)
            if not self.origin:
                self.W0 = self.rv([0.0])
        else:
            if self.origin:
                self.W = self.init
            else:
                self.W,self.W0 = self.init
                
        mistakes = 0
        if self.origin:
            for t in range(T):
                no_change = 0
                for i in range(n):
                    x_row = self.rv(X[:,i])
                    y = self.rv(Y[:,i])
                    test_stat = y*np.dot(x_row,self.W)
                    if test_stat[0,0] <= 0.0:
                        mistakes += 1
                        self.W += x_row.T*y
                        # print('W({0:d}):'.format(mistakes),W)
                        if self.hook: self.hook((self.W, self.rv([0.0])))
                    else:
                        no_change = no_change + 1
                if no_change == n: break
                self.epoch_loss[t] = n - self.score(X,Y,self.W,self.W0)
                if self.verbose:
                    self.status_bar(t,T)
        else:
            for t in range(T):
                no_change = 0
                for i in range(n):
                    x_row = self.rv(X[:,i])
                    y = self.rv(Y[:,i])
                    test_stat = y*(np.dot(x_row,self.W) + self.W0)
                    if test_stat[0,0] <= 0.0:
                        mistakes += 1
                        self.W += x_row.T*y
                        self.W0 += y
                        if self.hook: self.hook((self.W, self.W0))
                    else:
                        no_change = no_change + 1
                if no_change == n: break
                Ypred = self.predict(X)
                self.epoch_loss[t] = n - self.score(Ypred,Y)
                if self.verbose:
                    self.status_bar(t,T)
        if self.verbose:
            self.print_this('\nmistakes: {0:d}'.format(mistakes),self.logfile)
        return 

    
    def avg_perceptron(self,X,Y,params={}):
        """ Averaged Perceptron algorithm.
            X: (d,n) array for n data points in d dimensions
            Y: (1,n) array containing +-1.
            params: hyperparameter dictionary.
        """
        T = params.get('T',1000)
        self.epochs = np.arange(T)+1.0
        self.epoch_loss = np.zeros_like(self.epochs)
        
        d,n = X.shape
        if Y.shape != (1,n):
            raise TypeError('incompatible features and labels in avg_perceptron().')
            
        if self.init is None:
            self.W = self.cv([0.0]*d)
            self.W0 = self.rv([0.0])
        else:
            self.W,self.W0 = self.init

        Ws = 0.0*self.W
        W0s = 0.0*self.W0
        
        mistakes = 0
        for t in range(T):
            for i in range(n):
                x_row = self.rv(X[:,i])
                y = self.rv(Y[:,i])
                test_stat = y*(np.dot(x_row,self.W) + self.W0)
                if test_stat[0,0] <= 0.0:
                    mistakes = mistakes + 1
                    self.W += x_row.T*y
                    self.W0 += y
                Ws += self.W
                W0s += self.W0
            count = n*(t+1)
            self.W = Ws/count
            self.W0 = W0s/count
            Ypred = self.predict(X)
            self.epoch_loss[t] = n - self.score(Ypred,Y)
            if self.verbose:
                self.status_bar(t,T)
        return 

    def predict(self,X):
        return self.positive(X,self.W,self.W0)
    
#############################################

#############################################
class Module(object):
    def sgd_step(self,t,lrate):
        return # does nothing for modules without weights
    
#################################
# possible activation modules
#################
class Sigmoid(Module,MLUtilities):
    def __init__(self):
        self.net_type = 'class'
        
    def forward(self,Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self,dLdA):
        dLdZ = self.A*(1-self.A)*dLdA
        return dLdZ
    
    def predict(self):
        out = np.zeros_like(self.A)
        out[out > 0.5] = 1.0
        return out
#################

#################
class Tanh(Module,MLUtilities):
    def __init__(self):
        self.net_type = 'reg'
        
    def forward(self,Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self,dLdA):     
        return dLdA*(1-self.A**2)
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A)
#################


#################
class ReLU(Module,MLUtilities):
    def __init__(self):
        self.net_type = 'reg'
        
    def forward(self,Z):
        self.A = np.maximum(0.0,Z)
        return self.A

    def backward(self,dLdA):
        dLdZ = np.zeros_like(self.A)
        dLdZ[self.A > 0.0] = 1.0
        dLdZ *= dLdA
        return dLdZ
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A)
#################

#################
class Identity(Module,MLUtilities):
    def __init__(self):
        self.net_type = 'reg'
        
    def forward(self,Z):
        self.A = Z.copy()
        return self.A

    def backward(self,dLdA):
        return dLdA # dLdZ = dLdA when A = Z.
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A)
#################

#################
class SoftMax(Module,MLUtilities):
    def __init__(self):
        self.net_type = 'class'
        
    def forward(self,Z):
        exp_z = np.exp(Z) # (K,n_{sample})
        self.A = exp_z/np.sum(exp_z,axis=0)
        return self.A # (K,n_{sample})

    def backward(self,dLdA):
        dLdZ = self.A*(1-self.A)*dLdA # formally same as backward of Sigmoid()
        return dLdZ

    def predict(self):
        return np.array([np.argmax(self.A,axis=0)]).T # (n_{sample},1)
#################

#################################


#################################
# possible normalizations
#################
class DropNorm(Module,MLUtilities):
    def __init__(self,p_drop=0.2,rng=None):
        self.p_drop = p_drop
        self.rng = np.random.RandomState() if rng is None else rng

    def drop_fun(self,A):
        u = self.rng.rand(A.shape[0],A.shape[1])
        drop = np.ones_like(A)
        drop[u < self.p_drop] = 0.0 # since probab to drop is specified
        return A*drop
    
    def forward(self,A): # will always follow activation layer
        self.A = self.drop_fun(A) 
        return self.A # (K,n_{sample})

    def backward(self,dLdA):
        return dLdA # does nothing

#################################

#################################
class BatchNorm(Module,MLUtilities):
    # structure courtesy MIT-OLL IntroML Course
    def __init__(self,n_this,rng=None):
        self.n_this = n_this # input layer dimension
        self.rng = np.random.RandomState() if rng is None else rng
        self.eps = 1e-15
        self.G = rng.randn(n_this,1)/np.sqrt(n_this) # (n_this,1)
        self.B = np.zeros((n_this,1))#rng.randn(n_this,1) # (n_this,1)

    def forward(self,A):
        self.A = A
        self.K = A.shape[1] # mini-batch size
        
        self.mus = np.mean(A,axis=1,keepdims=True)
        self.vars = np.var(A,axis=1,keepdims=True)
        
        self.std_inv = 1/np.sqrt(self.vars+self.eps)
        self.A_min_mu = self.A-self.mus
        self.norm = self.A_min_mu/self.std_inv # normalised version of input

        return self.G*self.norm + self.B # scaled, shifted output

    def backward(self,dLdA):
        # dLdX_{ij} = dLdA_{ij}*G_i/(sig_i+eps)*[1 - (1/K)sum_q dLdA_{iq} - 1/(sig_i+eps)(1/K)sum_q dLdA_{iq}(A_{iq}-mu_i)(A_{ij}-mu_i)]
        dLdnorm = dLdA*self.G*self.std_inv # (n_this,K) * (n_this,1) = (n_this,K)
        dLdX = dLdnorm.copy() # first term
        dLdX -= np.mean(dLdnorm,axis=1,keepdims=True) # second term
        dLdX -= self.std_inv*self.A_min_mu*np.mean(dLdnorm*self.A_min_mu,axis=1,keepdims=True)

        self.dLdB = np.sum(dLdA,axis=1,keepdims=True)
        self.dLdG = np.sum(dLdA*self.norm,axis=1,keepdims=True)

        return dLdX

    def sgd_step(self,t,lrate):
        # adam support might be critical
        self.B -= lrate*self.dLdB
        self.G -= lrate*self.dLdG
        return 
#################################
    
#################################

#################################
# possible loss functions
#################
class NLL(Module,MLUtilities):
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (1,b)
        self.Y = Y
        self.Y[self.Y < 0] = 0.0 # ensure only 0 or 1 passed as Y
        Loss = -Y*np.log(Ypred) - (1-Y)*np.log(1-Ypred)
        return np.sum(Loss) # scalar

    def backward(self):
        dLdZ = (self.Ypred - self.Y) # (n_last,b)
        return dLdZ # normally would return dL/dA, but doing this will be numerically more stable when Ypred ~ 0.
#################

#################
class NLLM(Module,MLUtilities):
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (n_last,b)
        self.Y = Y # no check. user must ensure only integers 0..K-1 passed for K categories
        Loss = np.sum(-Y*np.log(Ypred),axis=0,keepdims=True) # (1,b)
        return np.sum(Loss) # (1,1)

    def backward(self):
        dLdZ = (self.Ypred - self.Y) # (n_last,b)
        return dLdZ # normally would return dL/dA, but doing this will be numerically more stable when Ypred ~ 0.
#################

#################
class Square(Module,MLUtilities):
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (n_last,b)
        self.Y = Y
        Loss = (Ypred - Y)**2
        return np.sum(Loss) # scalar (sum over dimensions gives dot products, then over samples gives total loss)

    def backward(self):
        dLdZ = 2*(self.Ypred - self.Y) # (n_last,b)
        return dLdZ
#################

#################
class Hinge(Module,MLUtilities):
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (n_last,b)
        self.Y = Y
        Loss = np.maximum(0.0,1-Ypred*Y)
        return np.sum(Loss) # scalar (expect n_last=1 in this case, then sum over samples gives total loss)

    def backward(self):
        dLdZ = np.zeros_like(self.Y)
        dLdZ[self.Ypred*self.Y < 1.0] = -1.0
        dLdZ *= self.Y  # (n_last,b)
        return dLdZ
#################

#################################
# Linear module
#################
class Linear(Module,MLUtilities):
    def __init__(self,n_this,n_next,rng=None,adam=True,B1_adam=0.9,B2_adam=0.999,eps_adam=1e-8):
        self.rng = np.random.RandomState() if rng is None else rng
        
        # input,output sizes
        self.n_this,self.n_next = (n_this,n_next)
        self.adam = adam
        self.B1_adam = B1_adam
        self.B2_adam = B2_adam
        self.eps_adam = eps_adam
        
        # initialize bias and weights
        self.W = rng.randn(n_this,n_next)/np.sqrt(n_this) # (n_this,n_next)
        self.W0 = rng.randn(n_next,1) # (n_next,1)
        if self.adam:
            self.M = np.zeros_like(self.W)
            self.M0 = np.zeros_like(self.W0)
            self.V = np.zeros_like(self.W)
            self.V0 = np.zeros_like(self.W0)
            
    def forward(self,A):
        self.A = A # (n_this,b) where b = batch size
        Z = np.dot(self.W.T,self.A) + self.W0 # (n_next,b)
        return Z

    def backward(self,dLdZ):
        # dLdZ (n_next,b)
        dLdA = np.dot(self.W,dLdZ) # (n_this,n_next).(n_next,b) = (n_this,b)

        self.dLdW = np.sum([np.dot(self.A[:,i:i+1],dLdZ[:,i:i+1].T) for i in range(self.A.shape[1])], axis=0) # (n_this,n_next)
        # !! FIND BETTER WAY OF DOING ABOVE STEP !! maybe np.tensordot?
        self.dLdW0 = np.sum(dLdZ,axis=-1,keepdims=True) # (n_next,1)
        if self.adam:
            self.M = self.B1_adam*self.M + (1-self.B1_adam)*self.dLdW
            self.V = self.B2_adam*self.V + (1-self.B2_adam)*self.dLdW**2
            self.M0 = self.B1_adam*self.M0 + (1-self.B1_adam)*self.dLdW0
            self.V0 = self.B2_adam*self.V0 + (1-self.B2_adam)*self.dLdW0**2
        
        return dLdA

    def sgd_step(self,t,lrate):
        if self.adam:
            corr_B1 = 1-self.B1_adam**(1+t) 
            corr_B2 = 1-self.B2_adam**(1+t)
            dW = self.M/corr_B1/np.sqrt(self.V/corr_B2 + self.eps_adam)
            dW0 = self.M0/corr_B1/np.sqrt(self.V0/corr_B2 + self.eps_adam)
        else:
            dW = self.dLdW
            dW0 = self.dLdW0
        self.W = self.W - lrate*dW
        self.W0 = self.W0 - lrate*dW0
        return 
#################################


#################################
# Main gradient descent (Sequential) module
#################
class Sequential(Module,MLUtilities,Utilities):
    def __init__(self,params={}):
        """ Main class to implement gradient descent for feed-forward neural network with error back-propagation.
            params should be dictionary with a subset of following keys:
            -- params['data_dim']: int, input data dimension
            -- params['L']: int, L >= 1, number of layers
            -- params['n_layer']: list of L int, number of units in each layer. ** must have n_layer[-1] = y.shape[0] **
            -- params['atypes']: list of L str, activation type in each layer chosen from ['sigm','tanh','relu','sm','lin']
            -- params['loss_type']: str, loss function in ['square','hinge','nll','nllm']
            -- params['neg_labels']: boolean, are actual labels {-1,+1} or {0,1} for binary classification. Default True ({-1,1}).
            ** Note that last entry in 'atypes' must be consistent with 'loss' **
            ** [{'square','hinge'} <-> 'lin','nll' <-> 'sigm','nllm' <-> 'sm'] **
            ** loss type will take precedence in case of inconsistency **
            -- params['adam']: boolean, whether or not to use adam in GD update (default True)
            -- params['reg_fun']: str, type of regularization.
                                  Accepted values ['bn','drop','none'] for batch-normalization, dropout or no reg, respectively.
                                  If 'drop', then value of 'p_drop' must be specified. Default 'none'.
            -- params['p_drop']: float between 0 and 1, drop probability.
                                 Only used if 'reg_fun' = 'drop'.
                                 Default value 0.5, but not clear if this is a good choice.
            -- params['seed']: int, random number seed.
            -- params['verbose']: boolean, whether of not to print output (default True).
            -- params['logfile']: None or str, file into which to print output (default None, print to stdout)

            Provides forward, backward, sgd_step and sgd methods. Use sgd to train on given data set.
        """

        self.n0 = params.get('data_dim',None)
        self.L = params.get('L',1)
        self.n_layer = params.get('n_layer',[1]) # last n_layer should be compatible with y.shape
        self.atypes = params.get('atypes',['sigm']) # default assumes classification problem
        self.loss_type = params.get('loss_type','nll') # loss and last atype must be consistent
        self.neg_labels = params.get('neg_labels',True)
        self.adam = params.get('adam',True)
        self.reg_fun = params.get('reg_fun','none')
        self.p_drop = params.get('p_drop',0.0)
        self.seed = params.get('seed',None)
        self.verbose = params.get('verbose',True)
        self.logfile = params.get('logfile',None)
        
        self.rng = np.random.RandomState(self.seed)

        if self.verbose:
            self.print_this("... setting up {0:d} layer feed-forward neural network".format(self.L),self.logfile)
        
        self.check_init()

        mod = [Linear(self.n0,self.n_layer[0],rng=self.rng,adam=self.adam)]
        for l in range(1,self.L+1):
            if self.atypes[l-1] == 'relu':
                mod.append(ReLU())
            elif self.atypes[l-1] == 'tanh':
                mod.append(Tanh())
            elif self.atypes[l-1] == 'sigm':
                mod.append(Sigmoid())
            elif self.atypes[l-1] == 'lin':
                mod.append(Identity())
            elif self.atypes[l-1] == 'sm':
                mod.append(SoftMax())
            if l < self.L:
                if self.reg_fun == 'drop':
                    mod.append(DropNorm(p_drop=self.p_drop,rng=self.rng))
                elif self.reg_fun == 'bn':
                    mod.append(BatchNorm(self.n_layer[l-1],rng=self.rng))
                mod.append(Linear(self.n_layer[l-1],self.n_layer[l],rng=self.rng,adam=self.adam))
                
        if self.verbose:
            self.print_this("... ... expecting data dim = {0:d}, target dim = {1:d}".format(self.n0,self.n_layer[-1]),self.logfile)
            self.print_this("... ... using hidden layers of sizes ["
                            +','.join([str(self.n_layer[i]) for i in range(self.L-1)])
                            +"]",self.logfile)
            self.print_this("... ... ... and activations ["
                            +','.join([self.atypes[i] for i in range(self.L-1)])
                            +"]",self.logfile)
            self.print_this("... ... using last activation layer '"+self.atypes[-1]+"'",self.logfile)
            self.print_this("... ... using loss function '"+self.loss_type+"'",self.logfile)
            if self.reg_fun == 'drop':
                self.print_this("... ... using drop regularization with p_drop = {0:.3f}".format(self.p_drop),self.logfile)
            elif self.reg_fun == 'bn':
                self.print_this("... ... using batch normalization",self.logfile)
            else:
                self.print_this("... ... not using any regularization",self.logfile)
            
        self.modules = mod
        self.net_type = 'reg' if self.loss_type == 'square' else 'class'
        self.modules[-1].net_type = self.net_type # set last activation module net_type

    def check_init(self):
        """ Run various self-consistency checks at initialization. """

        if self.n0 is None:
            raise ValueError("data_dim must be specified in Sequential()")
            
        if len(self.atypes) != self.L:
            raise TypeError('Incompatible atypes in Sequential(). Expecting size {0:d}, got {1:d}'.format(self.L,len(self.atypes)))
        
        if len(self.n_layer) != self.L:
            raise TypeError('Incompatible n_layer in Sequential(). Expecting size {0:d}, got {1:d}'.format(self.L,len(self.n_layer)))
        
        if self.loss_type in ['square','hinge']:
            self.loss = Square() if self.loss_type == 'square' else Hinge()
            if self.verbose & (self.atypes[-1] != 'lin'):
                self.print_this("Warning: last activation " + self.atypes[-1]
                                + " seems inconsistent with " + self.loss_type + " loss. Proceed with caution!",self.logfile)
        elif self.loss_type == 'nll':
            self.loss = NLL()
            if self.verbose & (self.atypes[-1] != 'sigm'):
                self.print_this("Warning: last activation " + self.atypes[-1]
                                + " seems inconsistent with " + self.loss_type + " loss. Proceed with caution!",self.logfile)
        elif self.loss_type == 'nllm':
            self.loss = NLLM()
            if self.verbose & (self.atypes[-1] != 'sm'):
                self.print_this("Warning: last activation " + self.atypes[-1]
                                + " seems inconsistent with " + self.loss_type + " loss. Proceed with caution!",self.logfile)
        else:
            raise ValueError("loss must be one of ['square','hinge','nll','nllm'] in Sequential().")

        if self.reg_fun not in ['bn','drop','none']:
            if self.verbose:
                print("reg_fun must be one of ['bn','drop','none'] in Sequential(). Setting to 'none'.")
            self.reg_fun = 'none' # safest is 'none' if user is trying something other than mini-batch.

    def forward(self,Xt): # update activations
        for m in self.modules:
            Xt = m.forward(Xt)
        return Xt

    def backward(self,delta): # update gradients
        for m in self.modules[-2::-1]:
            # reverse order, skip first module (i.e. last activation) since loss.backward gives dLdZ^last
            delta = m.backward(delta)
            
    def sgd_step(self,t,lrate): # update weights (GD update)
        for m in self.modules:
            m.sgd_step(t,lrate)
        return

    def train(self,X,Y,params={}):
        """ Main routine for training.
            Expect X.shape = (n0,n_samp), Y.shape = (n_layer[-1],n_samp)
        """
        max_epoch = params.get('max_epoch',100)
        lrate = params.get('lrate',0.005)
        mb_count = params.get('mb_count',1)
        
        if self.verbose:
            self.print_this("... training",self.logfile)
            
        d,n_samp = X.shape

        if d != self.n0:
            raise TypeError("Incompatible data dimension in Sequential.sgd(). Expecting {0:d}, got {1:d}".format(self.n0,d))
        if Y.shape[0] != self.n_layer[-1]:
            raise TypeError("Incompatible target dimension in Sequential.sgd(). Expecting {0:d}, got {1:d}"
                            .format(self.n_layer[-1],Y.shape[0]))
        if Y.shape[1] != n_samp:
            raise TypeError("Incompatible n_samp in data and target in Sequential.sgd().")

        if (mb_count > n_samp) | (mb_count < 1):
            if self.verbose:
                self.print_this("Incompatible mb_count in Sequential.sgd(). Setting to n_samp (standard SGD).",self.logfile)
            mb_count = n_samp
        if (mb_count < n_samp) & (mb_count > np.sqrt(n_samp)):
            if self.verbose:
                print_str = "Large mb_count might lead to uneven mini-batch sizes in Sequential.sgd()."
                print_str += " Setting to int(sqrt(n_samp))."
                self.print_this(print_str,self.logfile)
            mb_count = int(np.sqrt(n_samp))
            
        mb_size = n_samp // mb_count

        self.epochs = np.arange(max_epoch)+1.0
        self.epoch_loss = np.zeros(max_epoch)
        ind_shuff = np.arange(n_samp)
        for t in range(max_epoch):
            self.rng.shuffle(ind_shuff)
            for b in range(mb_count):
                sl = np.s_[b*mb_size:(b+1)*mb_size] if b < mb_count-1 else np.s_[b*mb_size:]                    
                data,target = X[:,sl].copy(),Y[:,sl].copy()

                Ypred = self.forward(data) # update activations. prediction for mini-batch

                batch_loss = self.loss.forward(Ypred,target) # calculate current batch loss, update self.loss
                self.epoch_loss[t] += batch_loss
                dLdZ = self.loss.backward() # loss.backward returns last dLdZ not dLdA

                self.backward(dLdZ) # update gradients
                # self.backward skips last activation, since loss.backward gives last dLdZ

                self.sgd_step(t,lrate) # gradient descent update
            if self.verbose:
                self.status_bar(t,max_epoch)

        # if self.reg_fun == 'drop':
        #     if self.verbose:
        #         self.print_this("... correcting for drop regularization",self.logfile)
        #     # multiply all weights by 1-p_drop. ** CHECK THIS ** (ML course says p, not 1-p!)
        #     # biases untouched.
        #     for m in self.modules[::3]: # note steps of 3 due to (Linear,Activation,DropNorm) repeating structure
        #         m.W *= (1-self.p_drop)
        # # commented out since forward method will always drop some activations when called by predict, so this would over-correct.
                
        if self.verbose:
            self.print_this("... ... done",self.logfile)
            
        return

    def predict(self,X):
        """ Predict targets for given data set. """
        if X.shape[0] != self.n0:
            raise TypeError("Incompatible data in Sequential.predict(). Expected {0:d}, got {1:d}".format(self.n0,X.shape[0]))
        # update all activations.
        self.forward(X)
        # modify last activation into labels if needed.
        # if labels, these will always be non-negative
        Ypred = self.modules[-1].predict()
        if (self.net_type == 'class') & self.neg_labels:
            # convert 0 to -1 if needed.
            Ypred[Ypred == 0.0] = -1.0
        return Ypred
        
#################################


#################################
# Wrapper to systematically develop NN
#################
class BuildNN(Module,MLUtilities,Utilities):
    """ Systematically build and train feed-forward NN for given set of data and targets. """
    def __init__(self,X=None,Y=None,train_frac=0.5,
                 min_layer=1,max_layer=6,max_ex=2,target_test_loss=1e-2,loss_type='square',neg_loss=True,
                 seed=None,verbose=True,logfile=None):
        self.X = X
        self.Y = Y
        self.train_frac = train_frac
        self.min_layer = min_layer # min no. of layers
        self.max_layer = max_layer # max no. of layers
        self.max_ex = max_ex # max number of extra dimensions (compared to data dimensions) in hidden layers
        self.target_test_loss = target_test_loss
        self.loss_type = loss_type
        self.neg_loss = neg_loss # in case of classification, are labels {-1,1} (True) or {0,1} (False)
        self.seed = seed
        self.verbose = verbose
        self.logfile = logfile

        if self.verbose:
            self.print_this("Building feed-forward neural network...",self.logfile)
        
        self.check_input()

        self.n_samp = self.X.shape[1]
        self.data_dim = self.X.shape[0]
        self.target_dim = self.Y.shape[0]            
        self.n_train = np.rint(self.train_frac*self.n_samp).astype(int)
        self.n_test = self.n_samp - self.n_train
        if self.verbose:
            self.print_this("... found data set of dimension {0:d} with targets of dimension {1:d}"
                            .format(self.data_dim,self.target_dim),self.logfile)
            self.print_this("... found {0:d} samples"
                            .format(self.n_samp),self.logfile)
            self.print_this("... fraction {0:.3f} ({1:d} samples) will be used for training"
                            .format(self.train_frac,self.n_train),self.logfile)
            self.print_this("... setting up training and test samples",self.logfile)
        
        self.rng = np.random.RandomState(self.seed)
        self.ind_train = self.rng.choice(self.n_samp,size=self.n_train,replace=False)
        self.ind_test = np.delete(np.arange(self.n_samp),self.ind_train) # Note ind_test is ordered although ind_train is randomised

        self.X_train = self.X[:,self.ind_train].copy()
        self.Y_train = self.Y[:,self.ind_train].copy()

        self.X_test = self.X[:,self.ind_test].copy()
        self.Y_test = self.Y[:,self.ind_test].copy()
        
        if self.verbose:
            self.print_this("... setup complete",self.logfile)

    def check_input(self):
        """ Utility to check input for BuildNN(). """
        if (self.X is None) | (self.Y is None):
            raise TypeError("BuildNN() needs data set X (d,n_samp) and Y (K,n_samp) to be specified.")
        
        # Expect X.shape = (n0,n_samp), Y.shape = (n_layer[-1],n_samp)
        if self.X.shape[1] != self.Y.shape[1]:
            raise TypeError('Incompatible data and targets in BuildNN().')
        # Expect loss_type in []
        if self.loss_type not in ['square','hinge','nll','nllm']:
            raise ValueError("loss must be one of ['square','hinge','nll','nllm'] in BuildNN().")

        # train_frac should be between 0 and 1, exclusive
        if (self.train_frac <= 0.0) | (self.train_frac >= 1.0):
            if self.verbose:
                print("Warning: train_frac should be strictly between 0 and 1 in BuildNN(). Setting to 0.5")
            self.train_frac = 0.5
            
        return

    def trainNN(self):
        """ Train various networks and select the one that minimizes test loss.
            Returns: 
            -- net: instance of Sequential 
            -- params_setup: dictionary of parameters used for building net
            -- params_train: dictionary of parameters used for training net
            -- mean_test_loss: mean test loss using final network
        """
        if self.verbose:
            self.print_this("Initiating search... ",self.logfile)

        last_atypes = ['lin','tanh','sigm'] if self.loss_type == 'square' else ['sigm','tanh','sm','lin']
        hidden_atypes = ['tanh','relu']
        reg_funs = ['none']#,'bn']
        layers = np.arange(self.min_layer,self.max_layer+1)
        max_epochs = np.array([300,1000,3000]) 
        # max_epochs = max_epochs.astype(int)
        # max_epochs[max_epochs > 3333] = 3333 # hard upper bound for now. 
        mb_counts = lambda mep: 20 if mep < 1000 else 10 #(10 if mep < 10000 else 5)
        lrates = np.array([0.001,0.003,0.01])
        
        pset = {'data_dim':self.data_dim,'loss_type':self.loss_type,'adam':True,'seed':self.seed,
                'verbose':False,'logfile':self.logfile,'neg_loss':self.neg_loss}
        ptrn = {}

        net = None
        params_setup = None
        params_train = None

        cnt_max = layers.size*max_epochs.size*len(reg_funs)*lrates.size*(self.max_ex+1)*len(last_atypes)*len(hidden_atypes) 
        cnt = 0
        mean_test_loss_this = 1e30
        mean_test_loss = 1e25
        if self.verbose:
            self.print_this("... cycling over {0:d} possible options".format(cnt_max),self.logfile)
        for ll in range(layers.size):
            L = layers[ll]
            pset['L'] = L
            # meps = np.array([max_epochs[ll]//3,max_epochs[ll],3*max_epochs[ll]])
            for mep in max_epochs:
                ptrn['max_epoch'] = mep
                ptrn['mb_count'] = mb_counts(mep)
                for lrate in lrates:
                    ptrn['lrate'] = lrate
                    for ex in range(self.max_ex+1): 
                        pset['n_layer'] = [self.data_dim+ex]*(L-1) + [self.target_dim]
                        for rf in reg_funs:
                            pset['reg_fun'] = rf
                            for last_atype in last_atypes:
                                for htype in hidden_atypes:
                                    pset['atypes'] = [htype]*(L-1) + [last_atype]                                
                                    net_this = Sequential(params=pset)
                                    net_this.train(self.X_train,self.Y_train,params=ptrn)
                                    Ypred_this = net_this.predict(self.X_test)
                                    mean_test_loss_this = net_this.loss.forward(Ypred_this,self.Y_test)/self.n_test
                                    if mean_test_loss_this < mean_test_loss:
                                        # store the current best network
                                        net = copy.deepcopy(net_this)
                                        params_setup = copy.deepcopy(pset)
                                        params_setup['verbose'] = self.verbose
                                        net.verbose = self.verbose
                                        params_train = copy.deepcopy(ptrn)
                                        # record current best mean test loss
                                        mean_test_loss = 1.0*mean_test_loss_this

                                    if mean_test_loss <= self.target_test_loss:
                                        if self.verbose:
                                            self.print_this("\n... achieved target test loss; breaking out",self.logfile)
                                        return net,params_setup,params_train,mean_test_loss

                                    if self.verbose:
                                        self.status_bar(cnt,cnt_max)
                                    cnt += 1

        # return last stored network and mean test loss
        return net,params_setup,params_train,mean_test_loss
#################################


#################################
# State Machine
#################
class StateMachine(MLUtilities,Utilities):
    """ Basic building block for state machine. """
    def __init__(self,f=None,g=None,verbose=True,logfile=None):
        self.verbose = verbose
        self.logfile = logfile

        if (f is None) & self.verbose:
            self.print_this("Transducer not specified in StateMachine(). Defaulting to identity.",self.logfile)
        if (g is None) & self.verbose:
            self.print_this("Output function not specified in StateMachine(). Defaulting to identity.",self.logfile)
        self.f = f if f is not None else lambda s,x: s
        self.g = g if g is not None else lambda s: s
        

    def execute(self,inputs,initial_state):
        """ Execute state machine for given input and initialisation."""
        s = initial_state
        out = []
        for x in inputs:
            s = self.f(s,x)
            out.append(self.g(s))
        return np.array(out)
        
#################################


#################################
# Markov Decision Process
#################
class MarkovDecisionProcess(MLUtilities,Utilities):
    """ Basic building block for Markov decision process. """
    def __init__(self,states,actions,reward,transition,verbose=True,logfile=None):
        Utilities.__init__(self)
        self.verbose = verbose
        self.logfile = logfile
        self.states = states
        self.actions = actions
        self.reward = np.vectorize(reward)
        self.transition = transition

    def value_func(self,policy,horizon=None,gamma=0.1):
        """ Calculate (finite horizon) value for given policy. 
            -- policy: function object mapping states to actions. 
            -- horizon: None or non-negative integer. 
                        If None (default), then calculate infinite-horizon result with discount gamma. 
            -- gamma: discount value in (0,1). Only used if horizon is None.
            Returns array of values of shape (len(self.states),).
        """
        if horizon is None:
            if (gamma < 0.0) | (gamma > 1.0):
                raise ValueError("Discount gamma should be between zero and unity for infinite-horizon value function.")
        vec_pol = np.vectorize(policy)
        policy_actions = vec_pol(self.states)
        if np.any(self.select_not_these(policy_actions,set(self.actions))):
            raise ValueError("Unexpected action detected. Try with another policy.")

        policy_rewards = self.reward(self.states,policy_actions)
        
        trans_matrix = np.zeros((len(self.states),len(self.states)))
        for s in range(len(self.states)):
            trans_matrix[s] = self.transition[policy_actions[s]][s] # set each row

        values = np.zeros(len(self.states))
        if horizon is None:
            # careful with inversion
            values = np.dot(np.inv(np.eye(len(self.states)) - gamma*trans_matrix),policy_rewards)
        else:
            for h in range(1,horizon+1):
                values = policy_rewards + np.dot(trans_matrix,values)

        return values

    def value_iteration(self,horizon=None,gamma=1.0,eps=1e-3,max_iter=1000):
        """ Calculate (finite horizon) optimal action-value function Q^{h}(s,a) using value iteration.
            -- horizon: None or non-negative integer. 
                        If None (default), then calculate infinite-horizon result with discount gamma. 
            -- gamma: discount value in (0,1). Set to 1 if horizon is finite integer.
            -- eps: small positive float, controls stopping criterion for infinite-horizon case. Only used if horizon is None.
            -- max_iter: maximum number of iterations
            Returns array of values of shape (len(self.actions),len(self.states)).
        """
        if horizon is None:
            if (gamma < 0.0) | (gamma > 1.0):
                raise ValueError("Discount gamma should be between zero and unity for infinite-horizon value function.")
        else:
            gamma = 1.0
        Q = np.zeros((len(self.actions),len(self.states)))
        
        if horizon == 0:
            return Q
        
        rewards = np.zeros_like(Q)
        for a in range(len(self.actions)):
            rewards[a] = self.reward(self.states,self.actions[a])
            
        if horizon is None:
            Q_prev = Q.copy() # store previous Q values for exit condition
        else:
            h = 0 # counter for finite-horizon case
            
        for i in range(max_iter):
            Qmax = np.max(Q,axis=0)
            for a in range(len(self.actions)):
                Q[a] = rewards[a] + gamma*np.dot(self.transition[self.actions[a]],Qmax)
            if horizon is None:
                if np.max(np.fabs(Q - Q_prev)) < eps:
                    return Q
                Q_prev = Q.copy()
            else:
                if h == horizon-1:
                    return Q
                h += 1

        return Q

    def optimal_policy(self,horizon=None,gamma=1.0,eps=1e-3,max_iter=1000):
        Q = self.value_iteration(horizon=horizon,gamma=gamma,eps=eps,max_iter=max_iter)
        pol_vec = []
        a_inds = np.argmax(Q,axis=0)
        for s in range(len(self.states)):
            pol_vec.append(self.actions[a_inds[s]])
        def opt_pol(state):
            s = np.where(state == self.states)[0][0]
            return pol_vec[s]
        return np.vectorize(opt_pol)
        
#################################
