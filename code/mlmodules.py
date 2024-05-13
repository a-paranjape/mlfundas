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
    def sgd_step(self,t,lrate):
        return

    def save(self):
        return

    def load(self):
        return
    
    # force these methods to be defined explicitly in each module
    def forward(self,A):
        raise NotImplementedError
    
    def backward(self,dLdA):
        raise NotImplementedError
    
#################################
# possible activation modules
#################
class Sigmoid(Module,MLUtilities):
    net_type = 'class'
    
    def forward(self,Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self,dLdA):
        dLdZ = self.A*(1-self.A)*dLdA
        return dLdZ
    
    def predict(self):
        out = np.zeros_like(self.A)
        out[self.A > 0.5] = 1.0
        return out
#################

#################
class Tanh(Module,MLUtilities):
    net_type = 'reg'
    
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
    net_type = 'reg'
    
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
    net_type = 'reg'
    
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
    net_type = 'class'
    
    def forward(self,Z):
        exp_z = np.exp(Z - np.max(Z)) # (K,n_{sample}) # subtract max inside exp to control overflows, cancels in A.
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
    def __init__(self,layer=1,p_drop=0.2,rng=None):
        self.layer = layer # for compatibility with weighted modules
        self.p_drop = p_drop
        self.rng = np.random.RandomState() if rng is None else rng
        self.layer = layer # useful for tracking save/read filenames
        self.is_norm = True

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

    def backward(self,dLdA):
        # dLdX_{ij} = dLdA_{ij}*G_i/(sig_i+eps)*[1 - (1/K)sum_q dLdA_{iq} - 1/(sig_i+eps)(1/K)sum_q dLdA_{iq}(A_{iq}-mu_i)(A_{ij}-mu_i)]
        dLdnorm = dLdA*self.W*self.std_inv # (n_this,K) * (n_this,1) = (n_this,K)
        dLdX = dLdnorm.copy() # first term
        dLdX -= np.mean(dLdnorm,axis=1,keepdims=True) # second term
        dLdX -= self.std_inv*self.A_min_mu*np.mean(dLdnorm*self.A_min_mu,axis=1,keepdims=True)

        self.dLdW0 = np.sum(dLdA,axis=1,keepdims=True)
        self.dLdW = np.sum(dLdA*self.norm,axis=1,keepdims=True)
        
        if self.adam:
            self.M = self.B1_adam*self.M + (1-self.B1_adam)*self.dLdW
            self.V = self.B2_adam*self.V + (1-self.B2_adam)*self.dLdW**2
            self.M0 = self.B1_adam*self.M0 + (1-self.B1_adam)*self.dLdW0
            self.V0 = self.B2_adam*self.V0 + (1-self.B2_adam)*self.dLdW0**2

        return dLdX

    def sgd_step(self,t,lrate):
        if self.adam:
            corr_B1 = 1-self.B1_adam**(1+t) 
            corr_B2 = 1-self.B2_adam**(1+t)
            dW = self.M/corr_B1/np.sqrt(self.V/corr_B2 + self.eps_adam)
            dW0 = self.M0/corr_B1/np.sqrt(self.V0/corr_B2 + self.eps_adam)
        else:
            dW = self.dLdW
            dW0 = self.dLdW0
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
        self.W = self.cv(np.loadtxt(file_W))
        self.W0 = self.cv(np.loadtxt(file_W0))
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
        Loss = -Y*np.log(Ypred + 1e-15) - (1-Y)*np.log(1-Ypred + 1e-15)
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
        Loss = np.sum(-Y*np.log(Ypred + 1e-15),axis=0,keepdims=True) # (1,b)
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
        self.W = np.loadtxt(file_W)
        self.W0 = np.loadtxt(file_W0)
        if self.n_next == 1:
            self.W = self.cv(self.W)
            self.W0 = self.rv([self.W0])
        else:
            self.W0 = self.cv(self.W0)
        return
#################################

