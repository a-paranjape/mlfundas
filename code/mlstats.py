import numpy as np
from scipy import linalg

from mllib import MLUtilities
from mlmodules import Module

from cobaya.likelihood import Likelihood

###############################################
class Chi2(Module,MLUtilities):
    ###########################################
    def __init__(self,params={}):
        """ Chi-squared calculator for general purpose usage, including mini-batch gradient descent schemes. """
        self.Y_full = params.get('Y_full',None)
        if self.Y_full is None:
            raise ValueError("Need to supply targets on full sample (training+validation) as key 'Y_full' in Chi2.")        
        
        cov_mat = params.get('cov_mat',None)
        self.cov_mat = np.eye(self.Y_full.shape[1]) if cov_mat is None else cov_mat
        if self.cov_mat.shape != (self.Y_full.shape[1],self.Y_full.shape[1]):
            raise ValueError('Incompatible cov_mat in Chi2.')
        if np.any(linalg.eigvals(self.cov_mat) <= 0.0):
            raise ValueError('Non-positive definite covariance matrix detected')
        # this is full cov mat, to be passed only once by user in initial parameter dictionary

        self.subset = params.get('subset',np.s_[:]) # slice or array of indices, used to extract (shuffled) training or validation elems.
        self.Y_sub = self.Y_full[:,self.subset].copy()
        self.n_sub = self.Y_sub.shape[1] # n_sub = n_samp for this data set

        self.cov_mat_sub = self.cov_mat[:,self.subset][self.subset,:].copy()
        self.L_sub = linalg.cholesky(self.cov_mat_sub,lower=True) # so C = L L^T
        
        self.slice_b = params.get('slice_b',np.s_[:]) # batch slice, indexed on self.subset
        self.batch_mask = np.zeros(self.n_sub)
        self.batch_mask[self.slice_b] = 1.0 # batch selection mask
        
        # loss = sum_{i,j} res_i Cinv_{ij} res_j
        #      = sum_{i=1}^{nsamp} res_i sum_{j=1}^{nsamp} Cinv_{ij} res_j
        #      = sum_{i=1}^{nsamp} Loss_i
        # where
        # Loss_i = res_i sum_{j=1}^{nsamp} Cinv_{ij} res_j = res_i Lvec_i
        #
        # note res_i = Ypred_i - Y_i = m_i - Y_i
        #
        # d batch_loss / d m_k = sum_{i in batch} d(res_i Lvec_i) / d m_k
        #                      = sum_{i in batch} Lvec_i d res_i / d m_k + res_i d Lvec_i / d m_k
        #                      = sum_{i in batch} Lvec_i delta_{ik} + res_i d/dm_k sum_j Cinv_{ij} res_j
        #                      = (Lvec_k if k in batch else 0) + res_i sum_j Cinv_{ij} d res_j / d m_k 
        #                      = (Lvec_k if k in batch else 0) + res_i sum_j Cinv_{ij} delta_{jk} 
        #                      = (Lvec_k if k in batch else 0) + sum_{i in batch} res_i Cinv_{ik}
    ###########################################
    
    ###########################################
    def forward(self,Ypred_sub):
        resid_sub = Ypred_sub - self.Y_sub # (1,n_samp)
        self.resid_b = resid_sub*self.batch_mask # (1,n_samp) with non-zero only in batch
        z = linalg.cho_solve((self.L_sub,True),resid_sub[0],check_finite=False) # solves (L L^T) z = resid or z = C^-1 residual, shape (nsamp,)
        self.Loss_vec = self.cv(z*self.batch_mask) # (n_samp,1) with non-zero only in batch
        Loss = np.dot(self.resid_b,self.Loss_vec) # (1,1)
        return Loss[0,0] # scalar
    ###########################################

    ###########################################
    def backward(self):
        dLdm = self.Loss_vec.T # (1,n_samp)
        z = self.rv(linalg.cho_solve((self.L_sub,True),self.resid_b[0],check_finite=False))
        # solves (L L^T) z^T = resid or z = (C^-1 resid)^T, shape (1,n_samp)
        dLdm += z
        return dLdm
    ###########################################


#########################################
class Chi2Like(Likelihood):
    X = None
    Y = None
    cov_mat = None
    #########################################
    def initialize(self):
        self.loss_params = {'Y_full':self.Y,'cov_mat':self.cov_mat}
        self.loss = Chi2(params=self.loss_params)
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        model = self.provider.get_model()
        chi2 = self.loss.forward(model) # ensure model has shape (1,n_samp)
        return -0.5*chi2
    #########################################

#########################################
