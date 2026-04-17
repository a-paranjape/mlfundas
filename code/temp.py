

#################################
# Optimize hyperparams and architecture using random search
#################
class HyperOpt(Module,MLUtilities,Utilities):
    """ Systematically build and train feed-forward NN for given set of data and targets. """
    #########################################
    def __init__(self,setup_dict):
        """ Random search over hyperparameters to optimize (ensemble of) NN instances. 
            setup_dict is a dictionary specifying training data, network family and search parameters,
            with keys being a subset of
            ------------
            :: mandatory
            ------------
            -- X: training sample features, shape ( n_input,nsamp)
            -- Y: training sample labels,   shape (n_output,nsamp)
            -- theta_dim: int; dimensionality of parameter space in BiSequential (not needed for other network families)
            ------------
            :: optional
            ------------
            -- family: str [default 'seq']; one of 'seq' (Sequential), 'biseq' (BiSequential), 'gan' (GAN)
            ------
            :: :: training sample
            ------
            -- train_frac: float (default 0.8); fraction of input samples to use for training+validation, remaining used for hyperparam/architecture comparison.
            -- val_frac: float (default 0.2); fraction of train_frac to use for early-stopping validation. Set to zero to switch off validation check. 
            -- loss_type: str (default 'square'); one of ['square','hinge','nll','nllm'] depending on problem type.
            -- neg_labels: bool (default True); are actual labels {-1,+1} (True) or {0,1} (False) for binary classification.
            ------
            :: :: training setup
            ------
            -- standardize_X: bool (default True); whether or not to standardize features.
            -- standardize_Y: bool (default True); whether or not to standardize labels.
            -- max_epoch: int (default 1000000); maximum number of training epochs
            -- check_after: int (default 300); epoch after which to activate validation (early stopping) checks. 
                            To swith off early stopping, set >= max_epoch.
            -- decay_norm: int (default 2); value of norm for weight decay, either 1 or 2.
            -- test_type: str (default 'perc'); one of 'perc' (residual percentiles) or 'mse' (mean squared error),
                          relevant for regression (square/hinge loss).
            -- seed: int or None (default); seed for random number generation. 
            -- file_stem: str (default 'net'); top-level path to store temporary files and final outputs.

            -- n_iter: int (default 3); number of iterations for each choice of hyperparams + architecture
            -- max_config: int (default 10); total number of distinct configurations to search over.
               ** Note: ** Total number of networks trained will be (n_iter * max_config)

            -- ensemble: bool (default False); whether or not to use ensemble of networks.
            -- ensemble_size: int (default 5); number of top networks to use in ensemble. Should not be larger than max_config. (Only used if ensemble is True.)
            -- parallel: bool (default False); whether or not to parallelize analysis of each configuration.
            -- nproc: int (default 4); number of concurrent processes to spawn. (Only used if parallel is True.)
            -- fixed_width: bool or None (default True)
                            True : each layer l has the same width W_l = W sampled from the range
                            False: each layer l has a width W_l sampled independently from the range
                            None: layer widths telescope down from max to min (similar to 'autoenc' behaviour of BuildNN)
            ------
            :: :: sampled parameters
            ------
            -- layers: range for number of layers
                       dict with structure 
                       {'min': int (default 1), 'max': int (default 3)}
            -- widths: range for layer width
                       dict with structure 
                       {'min': int (default 2), 'max': int (default 2)}
            -- lglrates: range for log10(learning rate)
                         dict with structure 
                         {'min': float (default -2.0), 'max': float (default -1.0)}
            -- wt_decays: range for weight decay
                          dict with structure 
                          {'min': float (default 0.0), 'max': float (default 0.0)} [default is no weight decay]
            -- thresholds: None (default) or range for classification thresholds (only needed for classification problems)
                           If not None, expect dict with structure 
                           {'min': float (e.g., 0.4), 'max': float (e.g., 0.6)}
                           None will default to 0.5.
            -- htypes: None (default) or list; hidden activation types (will be randomly sampled). 
                       None will default to ['relu','tanh'].
                       If not None, expect subset of ['tanh','relu','lrelu','splus','sin','requ']. 
            -- lrelu_slopes: None (default) or range for slopes of LReLU
                             If not None, expect dict with structure 
                             {'min': float (e.g., -1e-2), 'max': float (e.g., 1e-2)}
                             None will default to 1e-2.
            -- reg_funs: None (default) or list; regularization function types (will be randomly sampled). 
                         None will default to ['none'].
                         If not None, expect subset of ['bn','drop','none']. 
            -- p_drops: None (default) or range for drop probabilities (only needed if reg_funs contains 'drop')
                        If not None, expect dict with structure 
                        {'min': float (e.g., 0.4), 'max': float (e.g., 0.6)}
                        None will default to 0.5.
            -- dream_schedules: None (default) or list of dicts (to be randomly sampled) containing following params defining periodic 'dream' state during training
                -- dream_every:int; enter one dream epoch every 'dream_every' normal epochs (default 10, i.e. dream state is 10% of total training duration)
                -- corrupt_X:float; fraction of each input feature to be replaced by output of basis layer during dreaming (default 0.8)
                -- corrupt_Y:float; fraction of labels to be shuffled during dreaming (default 0.1)
            ------
            :: :: I/O
            -- verbose: bool (default True); whether or not to generate verbose output.
            -- logfile: str or None (default); path to pipe verbose output to. Defaults to stdout.
            ------
            ------------
        """
        Utilities.__init__(self)
        
        self.family_dict = {'seq':{'name':'Sequential','module':Sequential},
                            'biseq':{'name':'BiSequential','module':BiSequential},
                            'gan':{'name':'GAN','module':GAN}}

        self.htypes_superset = ['tanh','relu','lrelu','splus','sin','requ']
        self.reg_funs_superset = ['bn','drop','none']
        
        self.X = params.get('X',None)
        self.Y = params.get('Y',None)
        self.theta_dim = params.get('theta_dim',None)

        self.family = params.get('family','seq')
        
        self.train_frac = params.get('train_frac',0.8)
        self.val_frac = params.get('val_frac',0.2)
        self.loss_type = params.get('loss_type','square')
        self.neg_labels = params.get('neg_labels',True)

        self.standardize_X = params.get('standardize_X',True)
        self.standardize_Y = params.get('standardize_Y',True)
        self.max_epoch = params.get('max_epoch',1000000)
        self.check_after = params.get('check_after',300)
        self.decay_norm = params.get('decay_norm',2)
        self.test_type = params.get('test_type','perc')
        self.seed = params.get('seed',None)
        
        self.max_config = params.get('max_config',10)
        self.n_iter = params.get('n_iter',3)
        self.ensemble = params.get('ensemble',False)
        self.ensemble_size = params.get('ensemble_size',5) 
        self.parallel = params.get('parallel',False)
        self.nproc = params.get('nproc',4)

        self.verbose = params.get('verbose',True)
        self.logfile = params.get('logfile',None)

        self.layers = params.get('layers',{'min':1,'max':3})
        self.widths = params.get('widths',{'min':2,'max':2})
        self.fixed_width = params.get('fixed_width',True)
        self.lglrates = params.get('lglrates',{'min':-2.0,'max':-1.0})
        self.wt_decays = params.get('wt_decays',{'min':0.0,'max':0.0})
        self.thresholds = params.get('thresholds',None)
        self.htypes = params.get('htypes',None)
        self.lrelu_slopes = params.get('lrelu_slopes',None)
        self.reg_funs = params.get('reg_funs',None)
        self.p_drops = params.get('p_drops',None)
        dream_schedules = params.get('dream_schedules',None)
        self.dream_schedules = [None] if ((dream_schedules is None) | (self.family != 'seq')) else dream_schedules 
        
        self.file_stem = params.get('file_stem','net')        
        Path(self.file_stem).mkdir(parents=True,exist_ok=True) # folder to store temporary networks
        if self.ensemble:
            self.file_stem_ensemble = file_stem+'/ensemble'
            Path(self.file_stem_ensemble).mkdir(parents=True,exist_ok=True) # folder to store network ensemble members
        
        self.check_input()

        self.family_name = self.family_dict[self.family]['name']
        self.family_module = self.family_dict[self.family]['module']

        if self.verbose:
            self.print_this("--------------",self.logfile)
            self.print_this("Hyperparameter + architecture optimization",self.logfile)
            self.print_this("--------------",self.logfile)
            self.print_this("... neural network family: "+self.family_name,self.logfile)

        self.n_samp = self.X.shape[1]
        self.data_dim = self.X.shape[0]
        if self.family_name == 'BiSequential':
            self.data_dim -= self.theta_dim
        self.target_dim = self.Y.shape[0]
        self.n_train = np.rint(self.train_frac*self.n_samp).astype(int)
        self.n_test = self.n_samp - self.n_train
        if self.verbose:
            if self.family_name != 'BiSequential':
                self.print_this("... found data set of dimension {0:d} with targets of dimension {1:d}"
                                .format(self.data_dim,self.target_dim),self.logfile)
            else:
                self.print_this("... found data set of dimension = {0:d} (expecting n0a = {1:d}, n0w = {2:d}) with targets of dimension {3:d}"
                                .format(self.data_dim+self.theta_dim,self.data_dim,self.theta_dim,self.target_dim),self.logfile)
            self.print_this("... found {0:d} samples"
                            .format(self.n_samp),self.logfile)
            self.print_this("... fraction {0:.3f} ({1:d} samples) will be used for training"
                            .format(self.train_frac,self.n_train),self.logfile)
            self.print_this("... will search over {0:d} configurations".format(self.max_config),self.logfile)
            if self.ensemble:
                self.print_this("... will store best {0:d} networks in ensemble".format(self.ensemble_size),self.logfile)
            if self.loss_type == 'square':
                if self.test_type == 'perc':
                    self.print_this("... will use residual percentiles for hyperparameter comparison",self.logfile)
                else:
                    self.print_this("... will use mean squared error for hyperparameter comparison",self.logfile)
            else:
                self.print_this("... will use misclassification fraction for hyperparameter comparison",self.logfile)
            if self.wt_decays['max'] > 0.0:
                self.print_this("... weight decays will use norm {0:d}".format(self.decay_norm),self.logfile)
        
        self.rng = np.random.RandomState(self.seed)
        
        if self.verbose:
            self.print_this("... setup complete",self.logfile)
    #############################

    #############################
    def check_input(self):
        """ Utility to check input for HyperOpt. """

        if self.family not in self.family_dict.keys():
            raise ValueError("Unrecognised family identifier " + self.family + " in HyperOpt. Expecting one of ["
                             + ','.join([key for key in self.family_dict.keys()]) + "]")

        if self.X is None:
            raise TypeError("HyperOpt needs data set X (d,n_samp) to be specified.")

        if self.family not in ['gan']:
            if self.Y is None:
                raise TypeError("HyperOpt needs data set Y (K,n_samp) to be specified.")
            if self.X.shape[1] != self.Y.shape[1]:
                raise TypeError('Incompatible data and targets in HyperOpt.')

        if self.family == 'biseq':
            if self.theta_dim is None:
                raise TypeError("HyperOpt for BiSequential needs theta_dim to be specified.")
            if self.Y.shape[0] != 1:
                raise Exception("HyperOpt for BiSequential needs target dimension = 1.")

        # train_frac should be between 0 and 1, exclusive
        if (self.train_frac <= 0.0) | (self.train_frac >= 1.0):
            if self.verbose:
                print("Warning: train_frac should be strictly between 0 and 1 in HyperOpt. Setting to 0.8.")
            self.train_frac = 0.8

        # val_frac should be >= 0 and strictly < 1
        if (self.val_frac < 0.0) | (self.val_frac >= 1.0):
            if self.verbose:
                print("Warning: val_frac should be >= 0 and strictly < 1 in HyperOpt. Setting to 0.2.")
            self.train_frac = 0.2
        
        if self.loss_type not in ['square','hinge','nll','nllm']:
            raise ValueError("loss must be one of ['square','hinge','nll','nllm'] in HyperOpt.")
            
        if self.decay_norm not in [1,2]:
            if self.verbose:
                print("Warning!: decay_norm must be one of [1,2] in HyperOpt. Setting to 2.")
            self.decay_norm = 2 # safest is 2 if user is unsure about role of decay norm

        if (self.loss_type == 'square') & (self.test_type not in ['perc','mse']):
            if self.verbose:
                print("Warning: test_type for regression should be one of ['perc','mse'] in HyperOpt. Setting to 'perc'.")
            self.test_type = 'perc'

        if self.ensemble & (self.family_name != 'Sequential'):
            if self.verbose:
                print("Warning: ensemble currently only available for Sequential family in HyperOpt. Setting ensemble = False.")
            self.ensemble = False
            
        if self.ensemble & (self.ensemble_size > self.max_config):
            if self.verbose:
                print("Warning: ensemble_size should be <= max_config in HyperOpt. Setting to max_config = {0:d}.".format(self.max_config))
            self.ensemble_size = self.max_config


        self.check_dict(self.layers,'layers',int,1,np.inf)        
        self.check_dict(self.widths,'widths',int,1,np.inf)
        self.check_dict(self.lglrates,'lglrates',float,-np.inf,np.inf)
        self.check_dict(self.wt_decays,'wt_decays',float,0.0,np.inf)
        
        if self.thresholds is not None:
            self.check_dict(self.thresholds,'thresholds',float,0.0,1.0)
        else:
            # ensure well-defined sampling even when not requested
            self.thresholds = {'min':0.5,'max':0.5} 
        
        if self.htypes is not None:
            self.check_list(self.htypes,'htypes',self.htypes_superset)
        else:
            self.htypes = ['relu','tanh']
        if self.layers['max'] == 1:
            self.htypes = [None]
            
        if ('lrelu' in self.htypes) & (self.lrelu_slopes is not None):
            self.check_dict(self.lrelu_slopes,'lrelu_slopes',float,-1.0,1.0)
        else:
            # ensure well-defined sampling even when not requested
            self.lrelu_slopes = {'min':1e-2,'max':1e-2} 
            
        if self.reg_funs is not None:
            self.check_list(self.reg_funs,'reg_funs',self.reg_funs_superset)
        else:
            self.reg_funs = ['none']

        if ('drop' in self.reg_funs) & (self.p_drops is not None):
            self.check_dict(self.p_drops,'p_drops',float,0.0,1.0)
        else:
            # ensure well-defined sampling even when not requested
            self.p_drops = {'min':0.5,'max':0.5} 
            
        return
    #############################

    #############################
    def check_dict(self,test_dict,dict_name,key_type,min_min,max_max):
        """ Utility to check expected dictionary structures for HyperOpt. Called by HyperOpt.check_input(). 
            -- test_dict: object to be tested as min/max dictionary
            -- dict_name: str; name by which object is called
            -- key_type: data type of keys 'min','max'
            -- min_min: minimum value of test_dict['min']
            -- max_max: maximum value of test_dict['max'].
        """
        if not isinstance(test_dict,dict):
            raise TypeError(dict_name + " should be dict with keys 'min','max' in HyperOpt.")
        for key in ['min','max']:
            if key not in test_dict.keys():
                raise Exception("key '"+key+"' missing in dict "+dict_name+" in HyperOpt.")
        for key in ['min','max']:
            if not isinstance(test_dict[key],key_type):
                raise TypeError("key '"+key+"' of dict "+dict_name+" should be "+str(key_type)+" in HyperOpt.")

        if test_dict['min'] < min_min:
            if self.verbose:
                print("Warning: "+dict_name+"['min'] should be >= "+str(min_min)+" in HyperOpt. Setting to "+str(min_min))
            test_dict['min'] = min_min

        if test_dict['max'] > max_max:
            if self.verbose:
                print("Warning: "+dict_name+"['max'] should be <= "+str(max_max)+" in HyperOpt. Setting to "+str(max_max))
            test_dict['max'] = max_max
            
        if test_dict['max'] < test_dict['min']:
            if self.verbose:
                print("Warning: "+dict_name+"['max'] should be >= "+dict_name+"['min'] in HyperOpt. Setting to "+dict_name+"['min'] = "+str(test_dict['min']))
            test_dict['max'] = test_dict['min']
            
        return
    #############################

    #############################
    def check_list(self,test_list,list_name,superset):
        """ Utility to check expected list structures for HyperOpt. Called by HyperOpt.check_input(). 
            -- test_list: object to be tested as list
            -- list_name: str; name by which object is called
            -- superset: list which is the expected superset of test_list.
        """
        if not isinstance(test_list,list):
            raise TypeError("Expecting "+list_name+" to be a list in HyperOpt.")
        for elem in test_list:
            if elem not in superset:
                raise ValueError("Expecting "+list_name+" to be a subset of ["+','.join(superset)+"] in HyperOpt.")        
        return
    #############################
    
    #############################
    def gen_train(self):
        """ Convenience function to be able to repeatedly split input data into training and test samples. """
        ind_train = self.rng.choice(self.n_samp,size=self.n_train,replace=False)
        ind_test = np.delete(np.arange(self.n_samp),ind_train) # Note ind_test is ordered although ind_train is randomised

        X_train = self.X[:,ind_train].copy()
        X_test = self.X[:,ind_test].copy()

        if self.family_name not in ['GAN']:
            Y_train = self.Y[:,ind_train].copy()
            Y_test = self.Y[:,ind_test].copy()
        else:
            Y_train = None
            Y_test = None

        del ind_train,ind_test

        return X_train,Y_train,X_test,Y_test
    #############################

    #############################
    def queue_train(self,r,X_train,Y_train,X_test,Y_test,pset,ptrn,cnt_max,mdict):
        """ Convenience function for use with MLUtilities.run_processes(). Expect r >= 0."""
        
        pset['file_stem'] = pset['file_stem'] + '_r{0:d}'.format(r)
        
        net = self.family_module(params=pset)
        if self.family_name in ['GAN']:
            raise NotImplementedError()
            # SEE BELOW
            net.train(X_train,params=ptrn)
        else:
            net.train(X_train,Y_train,params=ptrn)

        # BELOW NEEDS TO BE MODIFIED FOR HANDLING GAN
        if net.net_type == 'reg':
            if self.test_type == 'perc':
                resid = net.predict(X_test)/(Y_test + 1e-15) - 1.0
                resid = resid.flatten()
                ts = 0.5*(np.percentile(resid,95) - np.percentile(resid,5))
            elif self.test_type == 'mse':
                ts = np.sum((net.predict(X_test) - Y_test)**2)/(Y_test.size + 1e-15)
                ts = np.sqrt(ts)
        else:
            if Y_test.shape[0] == 1:
                ts = np.where(np.rint(net.predict(X_test)) != np.rint(Y_test))[0].size/Y_test.shape[1]
            else:
                asmc = self.assess_multi_classification(net.predict(X_test),Y_test)
                ts = 1.0 - asmc['accuracy']
                asmc = None
            # this is fraction of predictions that are incorrect
            
        if not np.isfinite(ts):
            ts = 1e30
            
        # save this network (weights, setup dict and loss history) to file
        net.save()
        if self.family_name in ['GAN']:
            raise NotImplementedError()
        # !! Below will currently fail for GAN !! .. update GAN to include save_loss_history, update below to account for different naming convention for GAN loss.        
        imax_trn = np.where(net.training_loss > 0.0)[0][-1]
        imax_val = np.where(net.val_loss > 0.0)[0][-1] if net.val_loss.max() > 0.0 else 0 # val_loss may not have been computed
        imax = np.max([imax_trn,imax_val]) + 1
        net.epochs = net.epochs[:imax]
        net.training_loss = net.training_loss[:imax]
        net.val_loss = net.val_loss[:imax]
        net.save_loss_history() 
        
        mdict[r] = {'net':net,'teststat':ts,'ptrain':ptrn}
        
        if self.verbose:
            self.status_bar(r,cnt_max)
            
        return
    #############################

    #############################
    def optimize(self):
        """ Train various networks and select the one(s) that minimize)s( test loss.
            Returns: 
            -- net/neo: instance of self.family_module / NetworkEnsembleObject 
            -- params_train: dictionary of parameters used for training net
            -- mean_test_loss: mean test loss using final network
        """
        if self.verbose:
            self.print_this("Initiating search... ",self.logfile)

        ##############################
        if self.family_name in ['GAN']:
            raise NotImplementedError()
        # change below to adapt to GAN conventions
        if self.loss_type in ['square','hinge']:
            last_atype = 'lin'
        elif self.loss_type == 'nll':
            last_atype = 'sigm'
        elif self.loss_type == 'nllm':
            last_atype = 'sm'
        else:
            raise ValueError("loss_type must be in ['square','hinge','nll','nllm']")
        ##############################
        
        mb_count = int(np.sqrt(self.n_train)) 
        
        pset = {'data_dim':self.data_dim,'loss_type':self.loss_type,'adam':True,'seed':self.seed,'decay_norm':self.decay_norm,
                'standardize_X':self.standardize_X,'standardize_Y':self.standardize_Y,
                'file_stem':self.file_stem+'/net','verbose':False,'logfile':self.logfile,'neg_labels':self.neg_labels}            
        ptrn = {'max_epoch':self.max_epoch,'mb_count':mb_count,'val_frac':self.val_frac,'check_after':self.check_after,'dream_schedule':{}}
        if self.family_name == 'BiSequential':
            pset['theta_dim'] = self.theta_dim

        # LHC: [layers,widths,lglrates,wt_decays,lrelu_slopes,thresholds,p_drops] --> N_lhc in number
        # ** pay attention to behaviour of fixed_width:
        # -- if True, then LHC will behave as usual, with one width per config
        # -- if None, then sampled width will be interpreted as basis size and config will have telescoping widths determined by length and data dims
        # -- if False, then width will be inflated to *vector* with additional random samples: one value for each layer
        # discrete sampling for: {htypes,reg_funs,dream_schedules} --> N_discrete in number
        # strategy:
        # -- create LHC with max_config samples
        # -- create independent lists of length max_config (rng.choice with replacement) for each discretely sampled parameter
        # -- combine to generate max_config tasks
        # -- feed max_config tasks to the queue_train.

        param_mins = [self.layers['min'],self.widths['min'],
                      self.lglrates['min'],self.wt_decays['min'],self.lrelu_slopes['min'],self.thresholds['min'],self.p_drops['min']]
        param_maxs = [self.layers['max']+1,self.widths['max']+1, # note +1: samples will be float, then rounded down using int()
                      self.lglrates['max'],self.wt_decays['max'],self.lrelu_slopes['max'],self.thresholds['max'],self.p_drops['max']]
        N_lhc = len(param_mins)
        
        params = self.gen_latin_hypercube(Nsamp=self.max_config,dim=N_lhc,param_mins=param_mins,param_maxs=param_maxs,rng=self.rng) # (max_config,n_lhc)
        params[:,2] = 10**params[:,2] # convert lglrate to lrate

        sample_htype = self.rng.choice(self.htypes,size=self.max_config,replace=True)
        sample_rf = self.rng.choice(self.reg_funs,size=self.max_config,replace=True)
        sample_ds = self.rng.choice(self.dream_schedules,size=self.max_config,replace=True)
        
        cnt_max = self.n_iter*self.max_config
        
        if self.verbose:
            self.print_this("... cycling over {0:d} repetitions of {1:d} possible configurations"
                            .format(self.n_iter,self.max_config),self.logfile)
            self.print_this("... setting tasks",self.logfile)
            
        tasks = []
        for c in range(self.max_config):
            # order: [L,W,lrate,wt_decay,lrelu_slope,threshold,p_drop]
            L = int(params[c,0])
            pset['L'] = L
            
            W = int(params[c,1])
            if self.fixed_width is None:
                pset['n_layer'] = [int(self.data_dim*(self.data_dim/W)**(-np.log(l+1)/np.log(L))) for l in range(1,L)] 
            else:
                pset['n_layer'] = [W]*(L-1) if self.fixed_width else [W] + list(self.rng.randint(param_mins[1],high=param_maxs[1],size=L-2))
            pset['n_layer'] += [self.target_dim]
            
            ptrn['lrate'] = params[c,2]
            pset['wt_decay'] = params[c,3]
            
            pset['lrelu_slope'] = params[c,4]
            pset['threshold'] = params[c,5]
            pset['p_drop'] = params[c,6]

            pset['reg_fun'] = sample_rf[c]
            ptrn['dream_schedule'] = sample_ds[c]

            htype = sample_htype[c]
            pset['atypes'] = [last_atype] if htype is None else [htype]*(L-1) + [last_atype]
            
            for it in range(self.n_iter):
                X_train,Y_train,X_test,Y_test = self.gen_train() # sample training+test data
                tasks.append((X_train,Y_train,X_test,Y_test,copy.deepcopy(pset),copy.deepcopy(ptrn),cnt_max))

        # train networks
        if len(tasks) != cnt_max:
            raise Exception("Mismatched length of tasks and cnt_max.")
        if self.verbose:
            self.print_this("... training using {0:d} process(es)".format(self.nproc),self.logfile)
        all_nets = self.run_processes(tasks,self.queue_train,self.nproc)

        del tasks
        
        tsvals = np.array([all_nets[r]['teststat'] for r in range(cnt_max)])
        
        if self.ensemble:
            sorter = np.argsort(tsvals)
            sorter = sorter[:self.ensemble_size] # select requested number of best networks
                
            cnt = 0 # re-order labels by increasing teststat value
            for s in sorter:
                net_dict = copy.deepcopy(all_nets[s])
                net = copy.deepcopy(net_dict['net'])
                net.file_stem = self.file_stem_ensemble + '/net_r{0:d}'.format(cnt)
                net.params['file_stem'] = net.file_stem
                net.modules = gen_filestems(net.modules,net.file_stem)
                net.save()
                net.save_loss_history()

                del net_dict['net']
                # now net_dict = {'teststat':value,'ptrain':dict} for this network

                # unfortunate hack
                file_stem_old = self.file_stem 
                self.file_stem = net.file_stem
                self.save_train(net_dict) 
                self.file_stem = file_stem_old
                
                net_dict = None
                net = None
                cnt += 1

            del all_nets
            shutil.rmtree(self.file_stem+'/')
            gc.collect()
            
            if self.verbose:
                self.print_this("... defining and loading NetworkEnsembleObject",self.logfile)
            neo = NetworkEnsembleObject(ensemble_dir=self.file_stem_ensemble,verbose=self.verbose,logfile=self.logfile)
            neo.load()
                
            return neo
        else:
            if self.verbose:
                self.print_this("\n... identifying and saving best network and its teststat and training params",self.logfile)
            ind_best = np.argmin(tsvals)
            best_net = copy.deepcopy(all_nets[ind_best])
            net = copy.deepcopy(best_net['net'])
            net.file_stem = self.file_stem
            net.params['file_stem'] = self.file_stem
            net.modules = gen_filestems(net.modules,self.file_stem)
            net.save()
            net.save_loss_history()

            del best_net['net']
            # now best_net = {'teststat':value,'ptrain':dict} for best network
            self.save_train(best_net)
            params_train = best_net['ptrain']
            teststat = best_net['teststat']

            del all_nets
            shutil.rmtree(self.file_stem+'/')
            del best_net
            gc.collect()

            # return last stored network, training params and residual test statistic
            return net,params_train,teststat
    #############################

    #############################
    def save_train(self,net_dict):
        """ Save training params and best test stat to file. """
        with open(self.file_stem + '_train.pkl', 'wb') as f:
            pickle.dump(net_dict,f)
    #############################

    #############################
    def load_train(self):
        """ Load training params from file. """
        with open(self.file_stem + '_train.pkl', 'rb') as f:
            best_net = pickle.load(f)
            params_train = best_net['ptrain']
            teststat = best_net['teststat']
        return params_train,teststat
    #############################

    #############################
    def load(self):
        """ Load existing network / ensemble. """
        if Path(self.file_stem).is_dir():
            shutil.rmtree(self.file_stem+'/')
        if self.ensemble:
            neo = NetworkEnsembleObject(ensemble_dir=self.file_stem_ensemble,verbose=self.verbose,logfile=self.logfile)
            neo.load()
            return neo
        else:
            with open(self.file_stem + '.pkl', 'rb') as f:
                params_setup = pickle.load(f)
            net = self.family_module(params=params_setup)
            net.load()
            net.load_loss_history()
            return net
    #############################
#################################
