import numpy as np
from utilities import Utilities
from mllib import MLUtilities
from mlmodules import *
import copy
import pickle

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
        Utilities.__init__(self)
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
            -- params['atypes']: list of L str, activation type in each layer chosen from ['sigm','tanh','relu','sm','lin'] or 'custom...'.
                                 If 'custom...', then also define dictionary params['custom_atypes']
            -- params['custom_atypes']: dictionary with keys matching 'custom...' entry in params['atypes']
                                        with items being activation module instances.
            -- params['loss_type']: str, loss function in ['square','hinge','nll','nllm'] or 'custom...'.
                                    If 'custom...', then also define dictionary params['custom_loss']
            -- params['custom_loss']: dictionary with keys matching 'custom...' entry in params['loss_type']
                                      with items being loss module instances.
            -- params['neg_labels']: boolean, are actual labels {-1,+1} or {0,1} for binary classification. Default True ({-1,1}).
            ** Note that, ideally, last entry in 'atypes' must be consistent with 'loss' **
            ** [{'square','hinge'} <-> 'lin','nll' <-> 'sigm','nllm' <-> 'sm'] **
            -- params['standardize']: boolean, whether or not to standardize training data in train() (default True)
            -- params['adam']: boolean, whether or not to use adam in GD update (default True)
            -- params['wt_decay']: float, weight decay coefficient (should be non-negative; default 0.0)
            -- params['decay_norm']: int, norm of weight decay coefficient, either 2 or 1 (default 2)
            -- params['reg_fun']: str, type of regularization.
                                  Accepted values ['bn','drop','none'] for batch-normalization, dropout or no reg, respectively.
                                  If 'drop', then value of 'p_drop' must be specified. Default 'bn'.
            -- params['p_drop']: float between 0 and 1, drop probability.
                                 Only used if 'reg_fun' = 'drop'.
                                 Default value 0.5, but not clear if this is a good choice.
            -- params['seed']: int, random number seed.
            -- params['file_stem']: str, common stem for generating filenames for saving (should include full path).
            -- params['verbose']: boolean, whether of not to print output (default True).
            -- params['logfile']: None or str, file into which to print output (default None, print to stdout)

            Provides forward, backward, sgd_step and sgd methods. Use sgd to train on given data set.
        """
        Utilities.__init__(self)
        self.params = params
        self.n0 = params.get('data_dim',None)
        self.L = int(params.get('L',1))
        self.n_layer = params.get('n_layer',[1]) # last n_layer should be compatible with y.shape
        self.atypes = params.get('atypes',['sigm']) # default assumes classification problem
        self.custom_atypes = params.get('custom_atypes',None) 
        self.loss_type = params.get('loss_type','nll') # loss and last atype must be consistent
        self.custom_loss = params.get('custom_loss',None) 
        self.neg_labels = params.get('neg_labels',True)
        self.standardize = params.get('standardize',True)
        self.adam = params.get('adam',True)
        self.reg_fun = params.get('reg_fun','none')
        self.p_drop = params.get('p_drop',0.0)
        self.wt_decay = params.get('wt_decay',0.0)
        self.decay_norm = int(params.get('decay_norm',2))
        self.seed = params.get('seed',None)
        self.file_stem = params.get('file_stem','net')
        self.verbose = params.get('verbose',True)
        self.logfile = params.get('logfile',None)
        
        self.rng = np.random.RandomState(self.seed)
        
        self.Y_std = 1.0
        self.Y_mean = 0.0
        self.params['Y_std'] = self.Y_std
        self.params['Y_mean'] = self.Y_mean
        # will be reset by self.train() if self.standardize == True

        if self.verbose:
            self.print_this("... setting up {0:d} layer feed-forward neural network".format(self.L),self.logfile)
        
        self.check_init()
        
        # output of Modulator
        self.modules = Modulate(self.n0,self.n_layer,self.atypes,self.rng,self.adam,self.reg_fun,self.p_drop,self.custom_atypes)
                
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
            if self.wt_decay > 0.0:
                self.print_this("... ... using weight decay with coefficient {0:.3e} and norm {1:d}".format(self.wt_decay,self.decay_norm),
                                self.logfile)
            else:
                self.print_this("... ... not using any weight decay",self.logfile)
            
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
        elif self.loss_type[:6] == 'custom':
            if self.custom_loss is None:
                raise ValueError("Need to define dictionary custom_loss with keys containing "+self.loss_type)
            if self.loss_type not in list(self.custom_loss.keys()):
                raise ValueError("custom_loss keys must contain "+self.loss_type)
            self.loss = self.custom_loss[self.loss_type]
        else:
            raise ValueError("loss must be one of ['square','hinge','nll','nllm'] or 'custom...' in Sequential().")

        if self.standardize & (self.loss_type != 'square'):
            if self.verbose:
                self.print_this('Standardization incompatible with classification problems, switching off.',self.logfile)
            self.standardize = False

        for l in range(self.L):
            if self.atypes[l][:6] == 'custom':
                if self.custom_atypes is None:
                    raise ValueError("Need to define dictionary custom_atypes with keys containing "+self.atypes[l])
                if self.atypes[l] not in list(self.custom_atypes.keys()):
                    raise ValueError("custom_loss keys must contain "+self.atypes[l])

        if self.reg_fun not in ['bn','drop','none']:
            if self.verbose:
                print("reg_fun must be one of ['bn','drop','none'] in Sequential(). Setting to 'none'.")
            self.reg_fun = 'none' # safest is 'none' if user is trying something other than mini-batch.

        if self.wt_decay < 0.0:
            if self.verbose:
                print("wt_decay must be non-negative in Sequential(). Setting to zero.")
            self.wt_decay = 0.0 # safest is 0.0 if user is unsure about role of weight decay
            
        if self.decay_norm not in [1,2]:
            if self.verbose:
                print("decay_norm must be one of [1,2] in Sequential(). Setting to 2.")
            self.decay_norm = 2 # safest is 2 if user is unsure about role of decay norm

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
            m.sgd_step(t,lrate,wt_decay=self.wt_decay,decay_norm=self.decay_norm)
        return

    def calc_loss_decay(self):
        decay_loss = 0.0
        if self.decay_norm == 2:
            for m in self.modules:
                decay_loss += np.sum(m.W**2) if m.W is not None else 0.0
        elif self.decay_norm == 1:
            for m in self.modules:
                decay_loss += np.sum(np.fabs(m.W)) if m.W is not None else 0.0
        decay_loss *= self.wt_decay
        return decay_loss

    def train(self,X,Y,params={}):
        """ Main routine for training.
            Expect X.shape = (n0,n_samp), Y.shape = (n_layer[-1],n_samp)
        """
        max_epoch = params.get('max_epoch',100)
        lrate = params.get('lrate',0.005)
        mb_count = params.get('mb_count',1)
        val_frac = params.get('val_frac',0.2) # fraction of input data to use for validation
        check_after = params.get('check_after',10)
        
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

        n_val = np.rint(val_frac*n_samp).astype(int)
        n_samp -= n_val        

        if n_val > 0:
            ind_val = self.rng.choice(Y.shape[1],size=n_val,replace=False)
            ind_train = np.delete(np.arange(Y.shape[1]),ind_val) # Note ind_train is ordered although ind_val is randomised
        else:
            ind_train = np.arange(Y.shape[1])

        X_train = X[:,ind_train].copy()
        Y_train = Y[:,ind_train].copy()

        if n_val > 0:
            X_val = X[:,ind_val].copy()
            Y_val = Y[:,ind_val].copy()
        
        if self.standardize:
            self.Y_std = np.std(Y,axis=1)
            self.Y_mean = np.mean(Y,axis=1)
            self.params['Y_std'] = self.Y_std
            self.params['Y_mean'] = self.Y_mean
            Y_train -= self.Y_mean
            Y_train /= (self.Y_std + 1e-15)
            if n_val > 0:
                Y_val -= self.Y_mean
                Y_val /= (self.Y_std + 1e-15)
            
        if (mb_count > n_samp) | (mb_count < 1):
            if self.verbose:
                self.print_this("Incompatible mb_count in Sequential.sgd(). Setting to n_samp = {0:d} (standard SGD).".format(n_samp),
                                self.logfile)
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
        self.val_loss = np.zeros(max_epoch)
        ind_shuff = np.arange(n_samp)
        for t in range(max_epoch):
            self.rng.shuffle(ind_shuff)
            X_train_shuff = X_train[:,ind_shuff].copy()
            Y_train_shuff = Y_train[:,ind_shuff].copy()
            for b in range(mb_count):
                sl = np.s_[b*mb_size:(b+1)*mb_size] if b < mb_count-1 else np.s_[b*mb_size:]                    
                data,target = X_train_shuff[:,sl].copy(),Y_train_shuff[:,sl].copy()

                Ypred = self.forward(data) # update activations. prediction for mini-batch

                batch_loss = self.loss.forward(Ypred,target) # calculate current batch loss, update self.loss
                if self.wt_decay > 0.0:
                    batch_loss += self.calc_loss_decay()
                self.epoch_loss[t] += batch_loss
                dLdZ = self.loss.backward() # loss.backward returns last dLdZ not dLdA

                self.backward(dLdZ) # update gradients
                # self.backward skips last activation, since loss.backward gives last dLdZ

                self.sgd_step(t,lrate) # gradient descent update (will account for weight decay if requested)

            # validation check
            if n_val > 0:
                Ypred_val = self.forward(X_val) # update activations. prediction for validation data
                self.val_loss[t] = self.loss.forward(Ypred_val,Y_val) # calculate validation loss, update self.loss
                if t > check_after:
                    x = np.arange(t-check_after,t+1)
                    y = self.val_loss[x].copy()
                    # xbar = np.mean(x)
                    # slope = (np.mean(x*y)-xbar*np.mean(y))/(np.mean(x**2) - xbar**2 + 1e-15) # best fit slope
                    # chk_half = (self.val_loss[t] > 1.0*self.val_loss[t-check_after//2])
                    # chk = (self.val_loss[t-check_after//2] > 1.0*self.val_loss[t-check_after])
                    # if chk_half & chk:
                    if np.mean(x*y)-np.mean(x)*np.mean(y) > 0.0: # check for positive sign of best fit slope
                        if self.verbose:
                            self.print_this('',self.logfile)
                        break
            
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
        if self.standardize:
            # undo standardization
            Ypred *= (self.Y_std + 1e-15)
            Ypred += self.Y_mean
        if (self.net_type == 'class') & self.neg_labels:
            # convert 0 to -1 if needed.
            Ypred[Ypred < 1e-4] = -1.0
        return Ypred

    def save(self):
        """ Save current weights and setup params to file(s). """
        for m in self.modules:
            m.file_stem = self.file_stem
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.save()
        with open(self.file_stem + '.pkl', 'wb') as f:
            pickle.dump(self.params,f)
            
        return    

    # to be called after generating instance of Sequential() with correct setup params,
    # e.g. after invoking self.save() or after call to load method of BuildNN().
    def load(self):
        """ Load weights and setup params from file(s). """
        for m in self.modules:
            m.file_stem = self.file_stem
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.load()
        with open(self.file_stem + '.pkl', 'rb') as f:
            self.params = pickle.load(f)
            
        self.standardize = self.params['standardize']
        if self.standardize:
            self.Y_std = self.params['Y_std']
            self.Y_mean = self.params['Y_mean']
        
        return

    # to be called after generating/loading instance of Sequential() with correct setup params.
    def extract_basis(self):
        """ Extract penultimate layer of NN as collection of basis functions. """
        if self.verbose:
            self.print_this("... extracting basis functions",self.logfile)
        params_setup = copy.deepcopy(self.params)
        params_setup['L'] -= 1
        params_setup['n_layer'].pop(-1)
        params_setup['atypes'].pop(-1)
        params_setup['standardize'] = False # for safety. actually only Sequential.train and Sequential.load will give non-trivial effect of this key.
        basis = Sequential(params=params_setup)
        for m in range(len(basis.modules)):
            basis.modules[m] = copy.deepcopy(self.modules[m])
        return basis

    # to be called after invoking self.extract_basis()
    def combine_basis(self,basis,X):
        """ Simple wrapper to quickly test basis extraction by combining as per last layer of NN.
            -- basis: Sequential instance created by self.extract_basis
            -- X: input array of shape (n0,n_params) as used for NN (value of n_params can vary).
            Returns final output of NN as (activated) linear combination of basis functions.
        """
        basis_func = basis.predict(X)
        Z = self.modules[-2].forward(basis_func)
        A = self.modules[-1].forward(Z)
        if self.standardize:
            A *= self.Y_std
            A += self.Y_mean
        return A

    def calc_N_freeparams(self):
        """ Utility to calculate number of free parameters being optimized. """
        N = 0
        for l in range(self.L):
            n_lm1 = 1*self.n0 if l==0 else 1*self.n_layer[l-1]
            N += self.n_layer[l]*(n_lm1 + 1)
        return N
    
#################################


#################################
# Wrapper to systematically develop NN
#################
class BuildNN(Module,MLUtilities,Utilities):
    """ Systematically build and train feed-forward NN for given set of data and targets. """
    def __init__(self,X=None,Y=None,train_frac=0.5,val_frac=0.2,n_iter=3,standardize=True,
                 min_layer=1,max_layer=6,max_ex=2,target_test_stat=1e-2,loss_type='square',neg_labels=True,arch_type=None,wt_decays=[0.0],
                 seed=None,file_stem='net',verbose=True,logfile=None):
        Utilities.__init__(self)
        self.X = X
        self.Y = Y
        self.val_frac = val_frac
        self.train_frac = train_frac
        self.standardize = standardize
        self.n_iter = n_iter # no. of times to iterate training/test generation
        self.min_layer = min_layer # min no. of layers
        self.max_layer = max_layer # max no. of layers
        self.max_ex = max_ex # max number of extra dimensions (compared to data dimensions) in hidden layers
        self.max_ex_vals = np.arange(max_ex+1) if np.isscalar(max_ex) else self.max_ex
        self.target_test_stat = target_test_stat
        self.loss_type = loss_type
        self.neg_labels = neg_labels # in case of classification, are labels {-1,1} (True) or {0,1} (False)
        self.arch_type = arch_type # if not None, string describing architecture type to explore. currently accepts 'emulator'
        self.wt_decays = wt_decays
        self.seed = seed
        self.file_stem = file_stem
        self.verbose = verbose
        self.logfile = logfile

        if self.verbose:
            self.print_this("Feed-forward neural network setup...",self.logfile)
        
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

        if self.arch_type is not None:
            if self.arch_type not in ['emulator','no_reg']:
                raise ValueError("arch_type must be None or one of ['emulator','no_reg'] in BuildNN.")

        if len(self.wt_decays) < 1:
            if self.verbose:
                print("Warning: wt_decays should be non-empty list. Setting to [0.0]")
            self.wt_decays = 0.5            
            
        return

    def gen_train(self):
        """ Convenience function to be able to repeatedly split input data into training and test samples. """
        self.ind_train = self.rng.choice(self.n_samp,size=self.n_train,replace=False)
        self.ind_test = np.delete(np.arange(self.n_samp),self.ind_train) # Note ind_test is ordered although ind_train is randomised

        self.X_train = self.X[:,self.ind_train].copy()
        self.Y_train = self.Y[:,self.ind_train].copy()

        self.X_test = self.X[:,self.ind_test].copy()
        self.Y_test = self.Y[:,self.ind_test].copy()

        return

    def trainNN(self):
        """ Train various networks and select the one that minimizes test loss.
            Returns: 
            -- net: instance of Sequential 
            -- params_train: dictionary of parameters used for training net
            -- mean_test_loss: mean test loss using final network
        """
        if self.verbose:
            self.print_this("Initiating search... ",self.logfile)

        if self.loss_type in ['square','hinge']:
            last_atypes = ['lin','tanh'] #if self.arch_type != 'emulator' else ['lin','tanh']
        elif self.loss_type == 'nll':
            last_atypes = ['sigm','tanh']
        elif self.loss_type == 'nllm':
            last_atypes = ['sm']
        else:
            raise ValueError("loss_type must be in ['square','hinge','nll','nllm']")
        
        mb_count = int(np.sqrt(self.n_train)) #10
        max_epoch = 1000000 # validation checks will be active
        
        pset = {'data_dim':self.data_dim,'loss_type':self.loss_type,'adam':True,'seed':self.seed,'standardize':self.standardize,
                'file_stem':self.file_stem,'verbose':False,'logfile':self.logfile,'neg_labels':self.neg_labels}
        ptrn = {'max_epoch':max_epoch,'mb_count':mb_count,'val_frac':self.val_frac}
        
        if self.arch_type in [None,'no_reg']:
            reg_funs = ['none']
            if self.arch_type is None:
                reg_funs += ['bn']
            layers = np.arange(self.min_layer,self.max_layer+1)
            lrates = np.array([0.001,0.003,0.01,0.03,0.1]) #np.array([0.001,0.003,0.01])
            ptrn['check_after'] = 100
        elif self.arch_type == 'emulator':
            reg_funs = ['none']
            layers = np.array([4,8,12])
            lrates = np.array([1e-3,1e-4])
            ptrn['check_after'] = 300
            
        hidden_atypes = ['tanh','relu'] if layers.max() > 1 else [None]

        net = None
        params_setup = None
        params_train = None

        cnt_max = self.n_iter*layers.size*len(self.wt_decays)*len(reg_funs)*lrates.size*len(self.max_ex_vals)*len(last_atypes)*len(hidden_atypes) 
        cnt = 0
        ts_this = 1e30
        teststat = 1e25
        if self.verbose:
            self.print_this("... cycling over {0:d} repetitions of {1:d} possible options"
                            .format(self.n_iter,cnt_max//self.n_iter),self.logfile)
        # compare_Y = self.rv(np.ones(self.n_test))
        for wt_decay in self.wt_decays:
            pset['wt_decay'] = wt_decay
            for ll in range(layers.size):
                L = layers[ll]
                pset['L'] = L
                for lrate in lrates:
                    ptrn['lrate'] = lrate
                    for ex in self.max_ex_vals: 
                        pset['n_layer'] = [self.data_dim+ex]*(L-1) + [self.target_dim]
                        for rf in reg_funs:
                            pset['reg_fun'] = rf
                            for last_atype in last_atypes:
                                for htype in hidden_atypes:
                                    pset['atypes'] = [last_atype] if htype is None else [htype]*(L-1) + [last_atype]
                                    for it in range(self.n_iter):
                                        self.gen_train() # sample training+test data
                                        net_this = Sequential(params=pset)
                                        net_this.train(self.X_train,self.Y_train,params=ptrn)
                                        if net_this.net_type == 'reg':
                                            resid = net_this.predict(self.X_test)/(self.Y_test + 1e-15) - 1.0
                                            resid = resid.flatten()
                                            ts_this = 0.5*(np.percentile(resid,84) - np.percentile(resid,16))
                                        else:
                                            ts_this = np.where(net_this.predict(self.X_test) != self.Y_test)[0].size/self.Y_test.shape[1]
                                            # this is fraction of predictions that are incorrect 
                                        if not np.isfinite(ts_this):
                                            ts_this = 1e30
                                        if ts_this < teststat:
                                            # store the current best network
                                            net = copy.deepcopy(net_this)
                                            params_setup = copy.deepcopy(pset)
                                            params_setup['verbose'] = self.verbose
                                            net.verbose = self.verbose
                                            params_train = copy.deepcopy(ptrn)
                                            # record current best mean test loss
                                            teststat = 1.0*ts_this
                                            # save current best network (weights and setup + train dicts) to file
                                            net.save()
                                            self.save_train(params_train)

                                        if teststat <= self.target_test_stat:
                                            if self.verbose:
                                                self.print_this("\n... achieved target test loss; breaking out",self.logfile)
                                            return net,params_train,teststat

                                        if self.verbose:
                                            self.status_bar(cnt,cnt_max)
                                        cnt += 1

        # return last stored network, training params and residual test statistic
        return net,params_train,teststat

    def save_train(self,params_train):
        """ Save training params to file. """
        with open(self.file_stem + '_train.pkl', 'wb') as f:
            pickle.dump(params_train,f)

    def load_train(self):
        """ Load training params from file. """
        with open(self.file_stem + '_train.pkl', 'rb') as f:
            params_train = pickle.load(f)
        return params_train

    def load(self):
        """ Load existing network. """
        with open(self.file_stem + '.pkl', 'rb') as f:
            params_setup = pickle.load(f)
        net = Sequential(params=params_setup)
        net.load()
        return net
#################################




#################################
# BiSequential module for basis function training
#################
class BiSequential(Module,MLUtilities,Utilities):
    def __init__(self,params={}):
        """ Class to implement gradient descent for feed-forward neural network with error back-propagation,
            specialized for basis function extraction in function approximation problems.
            Two networks are trained, NNa and NNw for the basis functions and coefficients respectively.
            NNa takes input x and predicts a(x), NNw takes input theta and predicts w(theta), w0(theta),
            such that the binet predicts y(x,theta) = w(theta).a(x) + w0(theta)
            ** Currently ASSUMES function y(x,theta) being approximated is scalar. ** 
            params should be dictionary with a subset of following keys:
            ---- input
            -- params['data_dim']: int, input data (x) dimension
            -- params['theta_dim']: int, input parameters (theta) dimension

            ---- network NNa
            -- params['La']: int, La >= 1, number of layers in basis function NNa
            -- params['n_layer_a']: list of La int, number of units in each layer of NNa. ** must have n_layer_a[-1] >= 1 (no. of non-trivial basis funcs) **
            -- params['atypes_a']: list of La str, activation type in each layer chosen from ['sigm','tanh','relu','sm','lin'] or 'custom...'.
                                   If 'custom...', then also define dictionary params['custom_atypes_a']
            -- params['custom_atypes_a']: dictionary with keys matching 'custom...' entry in params['atypes_a']
                                          with items being activation module instances.
            -- params['wt_decay_a']: float, weight decay coefficient (should be non-negative; default 0.0)
            -- params['decay_norm_a']: int, norm of weight decay coefficient, either 2 or 1 (default 2)

            ---- network NNw
            -- params['Lw']: int, Lw >= 1, number of layers in coefficients NNw
            -- params['n_layer_w']: list of Lw int, number of units in each layer of NNw. ** must have n_layer_w[-1] = n_layer_a[-1]+1 **
            -- params['atypes_w']: list of Lw str, activation type in each layer chosen from ['sigm','tanh','relu','sm','lin'] or 'custom...'.
                                   If 'custom...', then also define dictionary params['custom_atypes_w']
            -- params['custom_atypes_w']: dictionary with keys matching 'custom...' entry in params['atypes_w']
                                          with items being activation module instances.
            -- params['wt_decay_w']: float, weight decay coefficient (should be non-negative; default 0.0)
            -- params['decay_norm_w']: int, norm of weight decay coefficient, either 2 or 1 (default 2)

            ---- common
            -- params['standardize']: boolean, whether or not to standardize training data in train() (default True)
            -- params['adam']: boolean, whether or not to use adam in GD update (default True)
            -- params['reg_fun']: str, type of regularization.
                                  Accepted values ['bn','drop','none'] for batch-normalization, dropout or no reg, respectively.
                                  If 'drop', then value of 'p_drop' must be specified. Default 'bn'.
            -- params['p_drop']: float between 0 and 1, drop probability.
                                 Only used if 'reg_fun' = 'drop'.
                                 Default value 0.5, but not clear if this is a good choice.
            -- params['seed']: int, random number seed.
            -- params['file_stem']: str, common stem for generating filenames for saving (should include full path).
            -- params['verbose']: boolean, whether of not to print output (default True).
            -- params['logfile']: None or str, file into which to print output (default None, print to stdout)

            Provides forward, backward, sgd_step and sgd methods. Use sgd to train on given data set.
        """
        Utilities.__init__(self)
        self.params = params
        
        self.n0a = params.get('data_dim',None)
        self.La = int(params.get('La',1))
        self.n_layer_a = params.get('n_layer_a',[1]) # last n_layer_a should be number of non-trivial basis functions
        self.atypes_a = params.get('atypes_a',['lin']) 
        self.custom_atypes_a = params.get('custom_atypes_a',None)
        self.wt_decay_a = params.get('wt_decay_a',0.0)
        self.decay_norm_a = int(params.get('decay_norm_a',2))

        self.n0w = params.get('theta_dim',None)
        self.Lw = int(params.get('Lw',1))
        self.n_layer_w = params.get('n_layer_w',[1]) # last n_layer_w should be 1 + number of non-trivial basis functions
        self.atypes_w = params.get('atypes_w',['lin']) 
        self.custom_atypes_w = params.get('custom_atypes_w',None)
        self.wt_decay_w = params.get('wt_decay_w',0.0)
        self.decay_norm_w = int(params.get('decay_norm_w',2))
        
        self.standardize = params.get('standardize',True)
        self.adam = params.get('adam',True)
        self.reg_fun = params.get('reg_fun','none')
        self.p_drop = params.get('p_drop',0.0)
        self.seed = params.get('seed',None)
        self.file_stem = params.get('file_stem','binet')
        self.verbose = params.get('verbose',True)
        self.logfile = params.get('logfile',None)
        
        self.rng = np.random.RandomState(self.seed)
        
        self.Y_std = 1.0
        self.Y_mean = 0.0
        self.params['Y_std'] = self.Y_std
        self.params['Y_mean'] = self.Y_mean
        # will be reset by self.train() if self.standardize == True

        if self.verbose:
            self.print_this("Setting up {0:d},{1:d} layer feed-forward bi-neural network".format(self.La,self.Lw),self.logfile)
            
        self.loss = Square() 
        self.net_type = 'reg'
        
        self.check_init()
        
        # output of Modulator
        self.modules_a = Modulate(self.n0a,self.n_layer_a,self.atypes_a,self.rng,self.adam,self.reg_fun,self.p_drop,self.custom_atypes_a)
        self.modules_w = Modulate(self.n0w,self.n_layer_w,self.atypes_w,self.rng,self.adam,self.reg_fun,self.p_drop,self.custom_atypes_w)
        
        # set last activation module net_type
        self.modules_a[-1].net_type = self.net_type 
        self.modules_w[-1].net_type = self.net_type
        
                
        if self.verbose:
            self.print_this("... ... expecting data dim = {0:d}, theta dim = {1:d}, output dim = 1".format(self.n0a,self.n0w),self.logfile)
            self.print_this("... ... NNa using hidden layers of sizes ["
                            +','.join([str(self.n_layer_a[i]) for i in range(self.La)])
                            +"]",self.logfile)
            self.print_this("... ... ... and activations ["
                            +','.join([self.atypes_a[i] for i in range(self.La)])
                            +"]",self.logfile)
            self.print_this("... ... NNw using hidden layers of sizes ["
                            +','.join([str(self.n_layer_w[i]) for i in range(self.Lw)])
                            +"]",self.logfile)
            self.print_this("... ... ... and activations ["
                            +','.join([self.atypes_w[i] for i in range(self.Lw)])
                            +"]",self.logfile)
            self.print_this("... ... using loss function 'square'",self.logfile)
            if self.reg_fun == 'drop':
                self.print_this("... ... using drop regularization with p_drop = {0:.3f}".format(self.p_drop),self.logfile)
            elif self.reg_fun == 'bn':
                self.print_this("... ... using batch normalization",self.logfile)
            else:
                self.print_this("... ... not using any regularization",self.logfile)
            if self.wt_decay_a > 0.0:
                self.print_this("... ... NNa using weight decay with coefficient {0:.3e} and norm {1:d}".format(self.wt_decay_a,self.decay_norm_a),
                                self.logfile)
            else:
                self.print_this("... ... not using any weight decay in NNa",self.logfile)
            if self.wt_decay_w > 0.0:
                self.print_this("... ... NNw using weight decay with coefficient {0:.3e} and norm {1:d}".format(self.wt_decay_w,self.decay_norm_w),
                                self.logfile)
            else:
                self.print_this("... ... not using any weight decay in NNw",self.logfile)


    def check_init(self):
        """ Run various self-consistency checks at initialization. """

        if self.n0a is None:
            raise ValueError("data_dim must be specified in BiSequential()")
        if self.n0w is None:
            raise ValueError("theta_dim must be specified in BiSequential()")
            
        if len(self.atypes_a) != self.La:
            raise TypeError('Incompatible atypes_a in BiSequential(). Expecting size {0:d}, got {1:d}'.format(self.La,len(self.atypes_a)))
        if len(self.atypes_w) != self.Lw:
            raise TypeError('Incompatible atypes_w in BiSequential(). Expecting size {0:d}, got {1:d}'.format(self.Lw,len(self.atypes_w)))
        
        if len(self.n_layer_a) != self.La:
            raise TypeError('Incompatible n_layer_a in BiSequential(). Expecting size {0:d}, got {1:d}'.format(self.La,len(self.n_layer_a)))
        if len(self.n_layer_w) != self.Lw:
            raise TypeError('Incompatible n_layer_w in BiSequential(). Expecting size {0:d}, got {1:d}'.format(self.Lw,len(self.n_layer_w)))

        if self.n_layer_w[-1] != self.n_layer_a[-1] + 1:
            raise Exception('Need n_layer_w[-1] = n_layer_a[-1]+1, found n_layer_w[-1]={0:d}, n_layer_a[-1]={1:d}'
                            .format(self.n_layer_w[-1],self.n_layer_a[-1]))
        
        for l in range(self.La):
            if self.atypes_a[l][:6] == 'custom':
                if self.custom_atypes_a is None:
                    raise ValueError("Need to define dictionary custom_atypes with keys containing "+self.atypes_a[l])
                if self.atypes_a[l] not in list(self.custom_atypes_a.keys()):
                    raise ValueError("custom_loss keys must contain "+self.atypes_a[l])
        for l in range(self.Lw):
            if self.atypes_w[l][:6] == 'custom':
                if self.custom_atypes_w is None:
                    raise ValueError("Need to define dictionary custom_atypes with keys containing "+self.atypes_w[l])
                if self.atypes_w[l] not in list(self.custom_atypes_w.keys()):
                    raise ValueError("custom_loss keys must contain "+self.atypes_w[l])

        if self.reg_fun not in ['bn','drop','none']:
            if self.verbose:
                print("reg_fun must be one of ['bn','drop','none'] in BiSequential(). Setting to 'none'.")
            self.reg_fun = 'none' # safest is 'none' if user is trying something other than mini-batch.

        if self.wt_decay_a < 0.0:
            if self.verbose:
                print("wt_decay_a must be non-negative in BiSequential(). Setting to zero.")
            self.wt_decay_a = 0.0 # safest is 0.0 if user is unsure about role of weight decay

        if self.wt_decay_w < 0.0:
            if self.verbose:
                print("wt_decay_w must be non-negative in BiSequential(). Setting to zero.")
            self.wt_decay_w = 0.0 # safest is 0.0 if user is unsure about role of weight decay
            
        if self.decay_norm_a not in [1,2]:
            if self.verbose:
                print("decay_norm_a must be one of [1,2] in BiSequential(). Setting to 2.")
            self.decay_norm_a = 2 # safest is 2 if user is unsure about role of decay norm
            
        if self.decay_norm_w not in [1,2]:
            if self.verbose:
                print("decay_norm_w must be one of [1,2] in BiSequential(). Setting to 2.")
            self.decay_norm_w = 2 # safest is 2 if user is unsure about role of decay norm
            

    def forward_a(self,Xt): # update activations
        for m in self.modules_a:
            Xt = m.forward(Xt)
        return Xt

    def forward_w(self,Xt): # update activations
        for m in self.modules_w:
            Xt = m.forward(Xt)
        return Xt

    def forward(self,X):
        apred = self.forward_a(X[self.n0w:,:])
        apred = np.concatenate((np.ones((1,apred.shape[1])),apred),axis=0) # (n_layer_w[-1],n_samp) 
        wpred = self.forward_w(X[:self.n0w,:])
        Ypred = np.sum(apred*wpred,axis=0,keepdims=True) # (1,n_samp)
        return Ypred,apred,wpred

    def backward_a(self,delta): # update gradients
        for m in self.modules_a[::-1]:
            # reverse order, expect first input = dL/dAa_La
            delta = m.backward(delta)

    def backward_w(self,delta): # update gradients
        for m in self.modules_w[::-1]:
            # reverse order, expect first input = dL/dAw_Lw
            delta = m.backward(delta)

    # choice 1: single routine performing both steps (with possibly different lrates).
    #           must be used in single batch of varying x,theta combinations
    def sgd_step(self,t,lrate_a,lrate_w): # update weights (GD update)
        for m in self.modules_a:
            m.sgd_step(t,lrate_a,wt_decay=self.wt_decay_a,decay_norm=self.decay_norm_a)
        for m in self.modules_w:
            m.sgd_step(t,lrate_w,wt_decay=self.wt_decay_w,decay_norm=self.decay_norm_w)
        return

    # # choice 2: separate routines for each sgd step. can be used in mini batches varying x and theta separately
    # def sgd_step_a(self,t,lrate): # update weights (GD update)
    #     for m in self.modules_a:
    #         m.sgd_step(t,lrate,wt_decay=self.wt_decay_a,decay_norm=self.decay_norm_a)
    #     return
    # def sgd_step_w(self,t,lrate): # update weights (GD update)
    #     for m in self.modules_w:
    #         m.sgd_step(t,lrate,wt_decay=self.wt_decay_w,decay_norm=self.decay_norm_w)
    #     return

    def calc_loss_decay(self):
        decay_loss_a = 0.0
        if self.decay_norm_a == 2:
            for m in self.modules_a:
                decay_loss_a += np.sum(m.W**2) if m.W is not None else 0.0
        elif self.decay_norm_a == 1:
            for m in self.modules_a:
                decay_loss_a += np.sum(np.fabs(m.W)) if m.W is not None else 0.0
        decay_loss_a *= self.wt_decay_a
        
        decay_loss_w = 0.0
        if self.decay_norm_w == 2:
            for m in self.modules_w:
                decay_loss_w += np.sum(m.W**2) if m.W is not None else 0.0
        elif self.decay_norm_w == 1:
            for m in self.modules_w:
                decay_loss_w += np.sum(np.fabs(m.W)) if m.W is not None else 0.0
        decay_loss_w *= self.wt_decay_w
        
        return decay_loss_a+decay_loss_w

    def train(self,X,Y,params={}):
        """ Main routine for training.
            Expect X.shape = (n0w+n0a,n_samp), Y.shape = (1,n_samp)
        """
        max_epoch = params.get('max_epoch',100)
        lrate_a = params.get('lrate_a',0.005)
        lrate_w = params.get('lrate_w',0.005)
        mb_count = params.get('mb_count',1)
        val_frac = params.get('val_frac',0.2) # fraction of input data to use for validation
        check_after = params.get('check_after',10)
        
        if self.verbose:
            self.print_this("... training",self.logfile)
            
        d,n_samp = X.shape

        if d != self.n0w+self.n0a:
            raise TypeError("Incompatible data dimension in BiSequential.train(). Expecting {0:d}+{1:d}, got {2:d}".format(self.n0w,self.n0a,d))
        if Y.shape[0] != 1:
            raise TypeError("Incompatible target dimension in BiSequential.train(). Expecting 1, got {0:d}"
                            .format(Y.shape[0]))
        if Y.shape[1] != n_samp:
            raise TypeError("Incompatible n_samp in data and target in BiSequential.train().")

        n_val = np.rint(val_frac*n_samp).astype(int)
        n_samp -= n_val        

        if n_val > 0:
            ind_val = self.rng.choice(Y.shape[1],size=n_val,replace=False)
            ind_train = np.delete(np.arange(Y.shape[1]),ind_val) # Note ind_train is ordered although ind_val is randomised
        else:
            ind_train = np.arange(Y.shape[1])

        X_train = X[:,ind_train].copy()
        Y_train = Y[:,ind_train].copy()

        if n_val > 0:
            X_val = X[:,ind_val].copy()
            Y_val = Y[:,ind_val].copy()
        
        if self.standardize:
            self.Y_std = np.std(Y,axis=1)
            self.Y_mean = np.mean(Y,axis=1)
            self.params['Y_std'] = self.Y_std
            self.params['Y_mean'] = self.Y_mean
            Y_train -= self.Y_mean
            Y_train /= (self.Y_std + 1e-15)
            if n_val > 0:
                Y_val -= self.Y_mean
                Y_val /= (self.Y_std + 1e-15)
            
        if (mb_count > n_samp) | (mb_count < 1):
            if self.verbose:
                self.print_this("Incompatible mb_count in BiSequential.sgd(). Setting to n_samp = {0:d} (standard SGD).".format(n_samp),
                                self.logfile)
            mb_count = n_samp
        if (mb_count < n_samp) & (mb_count > np.sqrt(n_samp)):
            if self.verbose:
                print_str = "Large mb_count might lead to uneven mini-batch sizes in BiSequential.sgd()."
                print_str += " Setting to int(sqrt(n_samp))."
                self.print_this(print_str,self.logfile)
            mb_count = int(np.sqrt(n_samp))
            
        mb_size = n_samp // mb_count

        self.epochs = np.arange(max_epoch)+1.0
        self.epoch_loss = np.zeros(max_epoch)
        self.val_loss = np.zeros(max_epoch)
        ind_shuff = np.arange(n_samp)
        for t in range(max_epoch):
            self.rng.shuffle(ind_shuff)
            X_train_shuff = X_train[:,ind_shuff].copy()
            Y_train_shuff = Y_train[:,ind_shuff].copy()
            for b in range(mb_count):
                sl = np.s_[b*mb_size:(b+1)*mb_size] if b < mb_count-1 else np.s_[b*mb_size:]                    
                data,target = X_train_shuff[:,sl].copy(),Y_train_shuff[:,sl].copy()

                # update activations. prediction for mini-batch
                Ypred,apred,wpred = self.forward(data)

                # loss calculation
                batch_loss = self.loss.forward(Ypred,target) # calculate current batch loss, update self.loss
                if (self.wt_decay_a > 0.0) | (self.wt_decay_w > 0.0):
                    batch_loss += self.calc_loss_decay()
                self.epoch_loss[t] += batch_loss

                # back-propagation
                dLdZ_last = self.loss.backward() # (1,b)
                dLdZ_a = dLdZ_last*wpred[1:,:] # multiply by coeffs of non-trivial basis funcs
                dLdZ_w = dLdZ_last*apred # multiply by basis funcs pre-pended by unity

                # update gradients
                self.backward_a(dLdZ_a) 
                self.backward_w(dLdZ_w) 

                self.sgd_step(t,lrate_a,lrate_w) # gradient descent update (will account for weight decay if requested)

            # validation check
            if n_val > 0:
                # update activations. prediction for validation data
                Ypred_val,apred_val,wpred_val = self.forward(X_val)
                self.val_loss[t] = self.loss.forward(Ypred_val,Y_val) # calculate validation loss, update self.loss
                if t > check_after:
                    x = np.arange(t-check_after,t+1)
                    y = self.val_loss[x].copy()
                    if np.mean(x*y)-np.mean(x)*np.mean(y) > 0.0: # check for positive sign of best fit slope
                        if self.verbose:
                            self.print_this('',self.logfile)
                        break
            
            if self.verbose:
                self.status_bar(t,max_epoch)
                
        if self.verbose:
            self.print_this("... ... done",self.logfile)
            
        return

    def predict(self,X):
        """ Predict targets for given data set. """
        if X.shape[0] != self.n0w+self.n0a:
            raise TypeError("Incompatible data in BiSequential.predict(). Expected {0:d}+{1:d}, got {2:d}".format(self.n0w+self.n0a,X.shape[0]))
        # update all activations and predict.
        Ypred,apred,wpred = self.forward(X)
        if self.standardize:
            # undo standardization
            Ypred *= (self.Y_std + 1e-15)
            Ypred += self.Y_mean
        return Ypred

    def save(self):
        """ Save current weights and setup params to file(s). """
        # NNa
        for m in self.modules_a:
            m.file_stem = self.file_stem + '_a'
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.save()
            
        # NNw
        for m in self.modules_w:
            m.file_stem = self.file_stem + '_w'
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.save()
            
        # params dict
        with open(self.file_stem + '.pkl', 'wb') as f:
            pickle.dump(self.params,f)
            
        return    

    # to be called after generating instance of BiSequential() with correct setup params, e.g. after invoking self.save() 
    def load(self):
        """ Load weights and setup params from file(s). """
        # NNa
        for m in self.modules_a:
            m.file_stem = self.file_stem + '_a'
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.load()
            
        # NNw
        for m in self.modules_w:
            m.file_stem = self.file_stem + '_w'
            if m.is_norm:
                m.file_stem += '_norm'
            m.file_stem += '_layer{0:d}'.format(m.layer)
            m.load()
            
        # params dict
        with open(self.file_stem + '.pkl', 'rb') as f:
            self.params = pickle.load(f)
            
        self.standardize = self.params['standardize']
        if self.standardize:
            self.Y_std = self.params['Y_std']
            self.Y_mean = self.params['Y_mean']
        
        return

    # to be called after generating/loading instance of BiSequential() with correct setup params.
    def extract_basis(self):
        """ Extract output of NNa as collection of basis functions. Returns instance of Sequential. """
        if self.verbose:
            self.print_this("... extracting basis functions from NNa",self.logfile)
        params_setup = copy.deepcopy(self.params)
        params_setup['L'] = params_setup['La']
        params_setup['n_layer'] = params_setup['n_layer_a']
        params_setup['atypes'] = params_setup['atypes_a']
        params_setup['standardize'] = False # for safety. actually only Sequential.train and Sequential.load will give non-trivial effect of this key. 
        basis = Sequential(params=params_setup) # note Sequential not BiSequential
        for m in range(len(basis.modules)):
            basis.modules[m] = copy.deepcopy(self.modules_a[m])
        return basis


    # to be called after generating/loading instance of BiSequential() with correct setup params.
    def extract_coeffs(self):
        """ Extract output of NNw as collection of coefficients. Returns instance of Sequential. """
        if self.verbose:
            self.print_this("... extracting coefficients from NNw",self.logfile)
        params_setup = copy.deepcopy(self.params)
        params_setup['L'] = params_setup['Lw']
        params_setup['n_layer'] = params_setup['n_layer_w']
        params_setup['atypes'] = params_setup['atypes_w']
        params_setup['standardize'] = False # for safety. actually only Sequential.train and Sequential.load will give non-trivial effect of this key. 
        coeffs = Sequential(params=params_setup) # note Sequential not BiSequential
        for m in range(len(coeffs.modules)):
            coeffs.modules[m] = copy.deepcopy(self.modules_w[m])
        return coeffs
    
    # # to be called after invoking self.extract_basis()
    # def combine_basis(self,basis,coeffs,X):
    #     """ Simple wrapper to quickly test basis extraction by combining as per last layers of NNa and NNw.
    #         -- basis: Sequential instance created by self.extract_basis
    #         -- coeffs: Sequential instance created by self.extract_coeffs
    #         -- X: input array of shape (n0,n_params) as used for NN (value of n_params can vary).
    #         Returns final output of NN as (activated) linear combination of basis functions.
    #     """
    #     raise NotImplementedError()
    #     # basis_func = basis.predict(X)
    #     # Z = self.modules[-2].forward(basis_func)
    #     # A = self.modules[-1].forward(Z)
    #     # if self.standardize:
    #     #     A *= self.Y_std
    #     #     A += self.Y_mean
    #     # return A

    def calc_N_freeparams(self):
        """ Utility to calculate number of free parameters being optimized. """
        N = 0
        for l in range(self.La):
            n_lm1 = 1*self.n0a if l==0 else 1*self.n_layer_a[l-1]
            N += self.n_layer_a[l]*(n_lm1 + 1)
            
        for l in range(self.Lw):
            n_lm1 = 1*self.n0w if l==0 else 1*self.n_layer_w[l-1]
            N += self.n_layer_w[l]*(n_lm1 + 1)
            
        return N
    
#################################
