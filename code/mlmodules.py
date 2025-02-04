import numpy as np
from mllib import MLUtilities

#############################################
class Module(object):
    def __init__(self,layer=1):
        # for compatibility with weighted modules 
        self.layer = layer
        self.W = None
        self.W0 = None
        
        # for saving / reading. will be dynamically modified.
        self.file_stem = 'net'
        self.is_norm = False # helps identify normalization modules
    
    # these methods do nothing for modules without weights    
    def sgd_step(self,t,lrate,wt_decay=0.0,decay_norm=2):
        return

    def save(self):
        return

    def load(self):
        return
    
    # force these methods to be defined explicitly in each module
    def forward(self,A):
        raise NotImplementedError
    
    def backward(self,dLdA,grad=False): 
        raise NotImplementedError
    
#################################
# possible activation modules
#################
class Sigmoid(Module,MLUtilities):
    net_type = 'class'
    threshold = 0.5
    
    def forward(self,Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self,dLdA,grad=False):
        # dLdA -> (n_this,b) if grad=False else (n_this,n_last,b)
        dLdZ = self.A*(1-self.A) # (n_this,b)
        if grad:
            dLdA = np.transpose(dLdA,axes=(1,0,2)) # (n_last,n_this,b)
        dLdZ = dLdZ*dLdA
        if grad:
            dLdZ = np.transpose(dLdZ,axes=(1,0,2)) # (n_this,n_last,b)
        return dLdZ
    
    def predict(self):
        # out = np.zeros_like(self.A)
        # out[self.A > self.threshold] = 1.0
        # return out
        return self.A if self.net_type == 'reg' else self.step_fun(self.A-self.threshold)
    
#################

#################
class Tanh(Module,MLUtilities):
    net_type = 'reg'
    threshold = 0.0
    
    def forward(self,Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self,dLdA,grad=None): 
        # dLdA -> (n_this,b) if grad=False else (n_this,n_last,b)
        if grad:
            dLdA = np.transpose(dLdA,axes=(1,0,2)) # (n_last,n_this,b)
        dLdZ = dLdA*(1-self.A**2)
        if grad:
            dLdZ = np.transpose(dLdZ,axes=(1,0,2)) # (n_this,n_last,b)
        return dLdZ
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A-self.threshold)
#################


#################
class ReLU(Module,MLUtilities):
    net_type = 'reg'
    threshold = 0.0
    
    def forward(self,Z):
        self.A = np.maximum(0.0,Z)
        return self.A

    def backward(self,dLdA,grad=None): 
        # dLdA -> (n_this,b) if grad=False else (n_this,n_last,b)
        dLdZ = np.ones_like(self.A)
        dLdZ[self.A <= 0.0] = 0.0
        if grad:
            dLdA = np.transpose(dLdA,axes=(1,0,2)) # (n_last,n_this,b)
        dLdZ = dLdZ*dLdA
        if grad:
            dLdZ = np.transpose(dLdZ,axes=(1,0,2)) # (n_this,n_last,b)
        return dLdZ
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A-self.threshold)
#################



#################
class LReLU(Module,MLUtilities):
    # leaky ReLU
    net_type = 'reg'
    threshold = 0.0
    slope = 1e-2 # -1 < slope < 1. slope = 0 is same as ReLU.
    
    def forward(self,Z):
        self.A = np.maximum(self.slope*Z,Z)
        return self.A

    def backward(self,dLdA,grad=None): 
        # dLdA -> (n_this,b) if grad=False else (n_this,n_last,b)
        dLdZ = np.ones_like(self.A)
        dLdZ[self.A <= 0.0] = self.slope
        if grad:
            dLdA = np.transpose(dLdA,axes=(1,0,2)) # (n_last,n_this,b)
        dLdZ = dLdZ*dLdA
        if grad:
            dLdZ = np.transpose(dLdZ,axes=(1,0,2)) # (n_this,n_last,b)
        return dLdZ
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A-self.threshold)
#################

#################
class Identity(Module,MLUtilities):
    net_type = 'reg'
    threshold = 0.0
    
    def forward(self,Z):
        self.A = Z.copy()
        return self.A

    def backward(self,dLdA,grad=False): 
        return dLdA # dLdZ = dLdA when A = Z.
    
    def predict(self):
        return self.A if self.net_type == 'reg' else self.step_fun(self.A-self.threshold)
#################

#################
class SoftMax(Module,MLUtilities):
    net_type = 'class'
    threshold = 0.5 # not used
    
    def forward(self,Z):
        exp_z = np.exp(Z - np.max(Z)) # (K,n_{sample}) # subtract max inside exp to control overflows, cancels in A.
        self.A = exp_z/np.sum(exp_z,axis=0)
        return self.A # (K,n_{sample})

    def backward(self,dLdA,grad=False): 
        # dLdA -> (K,b) if grad=False else (K,n_last,b)
        dLdZ = self.A*(1-self.A) # (n_this,b)
        if grad:
            dLdA = np.transpose(dLdA,axes=(1,0,2))
        dLdZ = dLdZ*dLdA
        if grad:
            dLdZ = np.transpose(dLdZ,axes=(1,0,2))
        # dLdZ = self.A*(1-self.A)*dLdA # formally same as backward of Sigmoid()
        return dLdZ

    def predict(self):
        return np.array([np.argmax(self.A,axis=0)]).T # (n_{sample},1)
#################

#################################


#################################
# possible normalizations
#################
class DropNorm(Module,MLUtilities):
    def __init__(self,layer=1,p_drop=0.2,rng=None,drop=True):
        self.layer = layer # for compatibility with weighted modules
        self.p_drop = p_drop
        self.drop = drop
        self.rng = np.random.RandomState() if rng is None else rng
        self.layer = layer # useful for tracking save/read filenames
        self.is_norm = True

    def drop_fun(self,A):
        self.u = self.rng.rand(A.shape[0],A.shape[1])
        drop = np.ones_like(A)
        drop[self.u < self.p_drop] = 0.0 # since probab to drop is specified
        return A*drop
    
    def forward(self,A): # will always follow activation layer
        self.A = self.drop_fun(A) if self.drop else A
        return self.A # (K,n_{sample})

    def backward(self,dLdA,grad=False):
        dLdZ = dLdA.copy()
        if not grad:
            if self.drop:
                dLdZ[self.u < self.p_drop] = 0.0 # zero out gradients for dropped units
        # else:
        #     pass # if gradient wrt input requested then do nothing
        return dLdZ

#################################

#################################
class BatchNorm(Module,MLUtilities):
    # structure courtesy MIT-OLL IntroML Course
    def __init__(self,n_this,layer=1,rng=None,adam=True,B1_adam=0.9,B2_adam=0.999,eps_adam=1e-8):
        self.n_this = n_this # input layer dimension
        self.rng = np.random.RandomState() if rng is None else rng
        self.eps = 1e-15
        self.W = rng.randn(n_this,1)/np.sqrt(n_this) # (n_this,1); called G in MIT-OLL course
        self.W0 = rng.randn(n_this,1) # (n_this,1); called B in MIT-OLL course
        self.layer = layer # useful for tracking save/read filenames        
        self.is_norm = True

        # adam support
        self.adam = adam
        self.B1_adam = B1_adam
        self.B2_adam = B2_adam
        self.eps_adam = eps_adam
        
        if self.adam:
            self.M = np.zeros_like(self.W)
            self.M0 = np.zeros_like(self.W0)
            self.V = np.zeros_like(self.W)
            self.V0 = np.zeros_like(self.W0)            

    def forward(self,A):
        self.A = A
        self.K = A.shape[1] # mini-batch size
        
        self.mus = np.mean(A,axis=1,keepdims=True)
        self.vars = np.var(A,axis=1,keepdims=True)
        
        self.std_inv = 1/np.sqrt(self.vars+self.eps)
        self.A_min_mu = self.A-self.mus
        self.norm = self.A_min_mu/self.std_inv # normalised version of input

        return self.W*self.norm + self.W0 # scaled, shifted output

    def backward(self,dLdA,grad=False):
        # dLdX_{ij} = dLdA_{ij}*G_i/(sig_i+eps)*[1 - (1/K)sum_q dLdA_{iq} - 1/(sig_i+eps)(1/K)sum_q dLdA_{iq}(A_{iq}-mu_i)(A_{ij}-mu_i)]
        dLdnorm = dLdA*self.W*self.std_inv # (n_this,K) * (n_this,1) = (n_this,K)
        dLdX = dLdnorm.copy() # first term
        dLdX -= np.mean(dLdnorm,axis=1,keepdims=True) # second term
        dLdX -= self.std_inv*self.A_min_mu*np.mean(dLdnorm*self.A_min_mu,axis=1,keepdims=True)

        if not grad:
            self.dLdW0 = np.sum(dLdA,axis=1,keepdims=True)
            self.dLdW = np.sum(dLdA*self.norm,axis=1,keepdims=True)

            if self.adam:
                self.M = self.B1_adam*self.M + (1-self.B1_adam)*self.dLdW
                self.V = self.B2_adam*self.V + (1-self.B2_adam)*self.dLdW**2
                self.M0 = self.B1_adam*self.M0 + (1-self.B1_adam)*self.dLdW0
                self.V0 = self.B2_adam*self.V0 + (1-self.B2_adam)*self.dLdW0**2
        # else:
        #     pass # if gradient wrt input requested then don't update derivatives wrt weights

        return dLdX

    def sgd_step(self,t,lrate,wt_decay=0.0,decay_norm=2):
        if self.adam:
            corr_B1 = 1-self.B1_adam**(1+t) 
            corr_B2 = 1-self.B2_adam**(1+t)
            dW = self.M/corr_B1/np.sqrt(self.V/corr_B2 + self.eps_adam)
            dW0 = self.M0/corr_B1/np.sqrt(self.V0/corr_B2 + self.eps_adam)
        else:
            dW = self.dLdW
            dW0 = self.dLdW0
        if wt_decay > 0.0:
            if decay_norm == 2:
                self.W *= (1 - lrate*wt_decay) # eqn 7.5 of DeepLearning book
            elif decay_norm == 1:
                self.W -= wt_decay*np.sign(self.W) # eqn 7.20 of DeepLearning book
        self.W -= lrate*dW
        self.W0 -= lrate*dW0
        return 

    def save(self):
        """ Save current weights to file(s). """
        file_W = self.file_stem + '_W.txt'
        file_W0 = self.file_stem + '_W0.txt' # simpler to keep these inside the method
        np.savetxt(file_W,self.W,fmt='%.10e')
        np.savetxt(file_W0,self.W0,fmt='%.10e')
        return    

    def load(self):
        """ Read weights from file(s). """
        file_W = self.file_stem + '_W.txt'
        file_W0 = self.file_stem + '_W0.txt' # simpler to keep these inside the method
        self.W = np.loadtxt(file_W).reshape(self.n_this,1)
        self.W0 = np.loadtxt(file_W0).reshape(self.n_this,1)
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
        self.Ypred[self.Ypred < 0] = 0.0 # ensure only 0 or 1 passed as Ypred
        Loss = -self.Y*np.log(self.Ypred + 1e-15) - (1-self.Y)*np.log(1-self.Ypred + 1e-15)
        return np.sum(Loss) # scalar

    def backward(self):
        dLdZ = (self.Ypred - self.Y) # (n_last,b)
        return dLdZ # normally would return dL/dA, but doing this will be numerically more stable when Ypred ~ 0.
#################

#################
class NLLM(Module,MLUtilities):
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (n_last,b), no check.
        self.Y = Y # no check. user must ensure only integers 0..K-1 passed for K categories
        Loss = np.sum(-self.Y*np.log(self.Ypred + 1e-15),axis=0,keepdims=True) # (1,b)
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
class Wasserstein(Module,MLUtilities):
    # for use in Earth Mover (Wasserstein-1) loss
    def forward(self,Ypred,Y):
        self.Ypred = Ypred # (1,b)
        self.Y = Y
        self.Y[self.Y < 0] = 0.0 # ensure only 0 or 1 passed as Y
        Loss = (2*self.Y - 1)*self.Ypred
        return np.sum(Loss) # scalar

    def backward(self):
        # sign will account for gradient ascent
        dLdA = -1.0*(2*self.Y - 1) # (n_last,b)
        return dLdA
#################

# #################
# class LossGAN(Module,MLUtilities):
#     def forward(self,D_x,D_Gz):
#         # expect D_x, D_Gz of shape (1,b) and continuous in (0,1)
#         Loss = np.mean(np.log(D_x.flatten() + 1e-15)) + np.mean(np.log(1-D_Gz.flatten() + 1e-15))
#         return Loss

#     def backward_dd(self,D_x):
#         # sign appropriate for ascent
#         dLdZdd = D_x - 1 # (1=n_last^D,b)
#         return dLdZdd/D_x.shape[-1]

#     def backward_dg(self,D_Gz):
#         # sign appropriate for ascent
#         dLdZdg = D_Gz # (1=n_last^D,b)
#         return dLdZdg/D_Gz.shape[-1]
    
#     def backward_g(self,D_Gz,Dprime_Gz):
#         # sign appropriate for ascent
#         dLdZg = -1.0*Dprime_Gz/(D_Gz + 1e-15) # (n0,1,b)
#         # # sign appropriate for descent
#         # dLdZg = -1.0*Dprime_Gz/(1-D_Gz + 1e-15) # (n0,1,b)
#         return np.squeeze(dLdZg,axis=1)/D_Gz.shape[-1] # (n0=n_last^G,b)
# #################

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
    def __init__(self,n_this,n_next,layer=1,rng=None,adam=True,B1_adam=0.9,B2_adam=0.999,eps_adam=1e-8):
        self.rng = np.random.RandomState() if rng is None else rng
        
        # input,output sizes
        self.n_this,self.n_next = (n_this,n_next)

        self.layer = layer # useful for tracking save/read filenames
        self.is_norm = False

        # adam support
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

    def backward(self,dLdZ,grad=False):
        # dLdZ (n_next,b) [or (n_next,n_last,b) if grad=True]
        dLdA = np.tensordot(self.W,dLdZ,axes=1) # (n_this,n_next).(n_next[,n_last],b) = (n_this[,n_last],b)

        if not grad:
            # self.dLdW = np.sum([np.dot(self.A[:,i:i+1],dLdZ[:,i:i+1].T) for i in range(self.A.shape[1])], axis=0) # (n_this,n_next)
            self.dLdW = np.tensordot(self.A,dLdZ.T,axes=1)
            self.dLdW0 = np.sum(dLdZ,axis=-1,keepdims=True) # (n_next,1)
            if self.adam:
                self.M = self.B1_adam*self.M + (1-self.B1_adam)*self.dLdW
                self.V = self.B2_adam*self.V + (1-self.B2_adam)*self.dLdW**2
                self.M0 = self.B1_adam*self.M0 + (1-self.B1_adam)*self.dLdW0
                self.V0 = self.B2_adam*self.V0 + (1-self.B2_adam)*self.dLdW0**2
        # else:
        #     pass # if gradient wrt input requested then don't update derivatives wrt weights
        
        return dLdA

    def sgd_step(self,t,lrate,wt_decay=0.0,decay_norm=2):
        if self.adam:
            corr_B1 = 1-self.B1_adam**(1+t) 
            corr_B2 = 1-self.B2_adam**(1+t)
            dW = self.M/corr_B1/np.sqrt(self.V/corr_B2 + self.eps_adam)
            dW0 = self.M0/corr_B1/np.sqrt(self.V0/corr_B2 + self.eps_adam)
        else:
            dW = self.dLdW
            dW0 = self.dLdW0
        if wt_decay > 0.0:
            if decay_norm == 2:
                self.W *= (1 - lrate*wt_decay) # eqn 7.5 of DeepLearning book
            elif decay_norm == 1:
                self.W -= wt_decay*np.sign(self.W) # eqn 7.20 of DeepLearning book
        self.W -= lrate*dW
        self.W0 -= lrate*dW0
        return 

    def save(self):
        """ Save current weights to file(s). """
        file_W = self.file_stem + '_W.txt'
        file_W0 = self.file_stem + '_W0.txt' # simpler to keep these inside the method
        np.savetxt(file_W,self.W,fmt='%.10e')
        np.savetxt(file_W0,self.W0,fmt='%.10e')
        return    

    def load(self):
        """ Read weights from file(s). """
        file_W = self.file_stem + '_W.txt'
        file_W0 = self.file_stem + '_W0.txt' # simpler to keep these inside the method
        self.W = np.loadtxt(file_W).reshape(self.n_this,self.n_next)
        self.W0 = np.loadtxt(file_W0).reshape(self.n_next,1)
        return
#################################

#################################
def Modulate(n0,n_layer,atypes,rng,adam,reg_fun,p_drop,custom_atypes,threshold,lrelu_slope=1e-2):
    """ Simple utility to produce modules for use in feed-forward networks. 
        Assumes all inputs have been checked. 
        Returns list of instantiated modules.
    """
    L = len(atypes)
    mod = [Linear(n0,n_layer[0],rng=rng,adam=adam,layer=1)]
    for l in range(1,L+1):
        if atypes[l-1] == 'relu':
            mod.append(ReLU(layer=l+1))
        elif atypes[l-1] == 'lrelu':
            mod_lrelu = LReLU(layer=l+1)
            mod_lrelu.slope = lrelu_slope
            mod.append(mod_lrelu)
        elif atypes[l-1] == 'tanh':
            mod.append(Tanh(layer=l+1))
        elif atypes[l-1] == 'sigm':
            mod.append(Sigmoid(layer=l+1))
        elif atypes[l-1] == 'lin':
            mod.append(Identity(layer=l+1))
        elif atypes[l-1] == 'sm':
            mod.append(SoftMax(layer=l+1))
        elif atypes[l-1][:6] == 'custom':
            mod.append(custom_atypes[atypes[l-1]])

        if l < L:
            if reg_fun == 'drop':
                mod.append(DropNorm(p_drop=p_drop,rng=rng,layer=l+1))
            elif reg_fun == 'bn':
                mod.append(BatchNorm(n_layer[l-1],rng=rng,adam=adam,layer=l+1))
            mod.append(Linear(n_layer[l-1],n_layer[l],rng=rng,adam=adam,layer=l+1))
        elif threshold is not None:
            mod[-1].threshold = threshold

    return mod
#################################
