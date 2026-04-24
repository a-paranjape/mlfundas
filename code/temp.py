
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

#############################################
class IllustrateNetwork(Utilities):
    """ Illustrate architecture of an existing network or network ensemble. """
    ###################
    def __init__(self,base_dir,file_stem,ensemble,family,verbose=True,logfile=None):
        """ Load existing network or network ensemble and create simple graphical illustration of its architecture.
            -- base_dir: str, path/to/folder where network (ensemble) was originally created from.
            -- file_stem: str, file stem (relative to base_dir) used for defining location of network (ensemble).
            -- ensemble: bool, whether location is an ensemble (True) or single network (False).
            -- family: str; one of 'seq' (Sequential), 'biseq' (BiSequential), 'gan' (GAN)
            -- verbose,logfile: usual I/O control
            Methods:
            -- 
        """
        Utilities.__init__(self)
        self.base_dir = base_dir
        self.ensemble = ensemble
        self.verbose = verbose
        self.logfile = logfile

        self.golden_inv = 2/(1+np.sqrt(5))
        
        setup_dict = {}
        setup_dict['X'] = np.zeros((1,10)) # dummy arrays
        setup_dict['Y'] = np.zeros((1,10)) # for HyperOpt
        setup_dict['verbose'] = False
        setup_dict['ensemble'] = self.ensemble
        setup_dict['file_stem'] = file_stem
        setup_dict['family'] = family
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir) # temporarily change to base dir
            self.hopt = HyperOpt(setup_dict=setup_dict)
            if ensemble:
                self.neo = self.hopt.load()
            else:
                self.net = self.hopt.load()
                self.params_train,bts = self.hopt.load_train()
        finally:
            os.chdir(original_dir) # change back to original dir
        
        return
    ###################

    ###################
    def illustrate(self,out_stem='./',out_name='',file_ext='png'):
        """ Create illustration of loaded network (ensemble). 
            -- out_stem: str, /path/to/output/folder/ (default cwd). 
            -- out_name: str, optional prefix for output filename (default '', will create file 'illustration.'+file_ext).
            -- file_ext: str, output file extension (default 'png')
        """
        Path(out_stem).mkdir(parents=True,exist_ok=True) # folder to store figure
        
        if self.ensemble:
            pass
        else:
            fig,ax = plt.subplots(1)
            self.illustrate_single(self.net,ax)

        separator = '' if out_stem[-1] == '/' else '/'
        outfile = out_stem + separator + out_name + 'illustration.' + file_ext
        if self.verbose:
            self.print_this("Writing to file: "+outfile,self.logfile)
        plt.savefig(outfile,bbox_inches='tight')
            
        return
    ###################


    ###################
    def illustrate_single(self,params,ax):
        """ Create illustration object of single network. 
            -- params: setup params dict of desired single network
            --  ax: matplotlib axis object on which to draw
        """
        bdry = np.array([[0,0],[0,self.golden_inv],
                         [1,0],[1,self.golden_inv]]) # define boundary
        bdry = bdry.T
        ax.plot(bdry[0],bdry[1],c='w') # set boundary
        ax.set_xticks([])              # kill tick marks
        ax.set_yticks([])
        rect_bdry = Rectangle((0,0),1,1,lw=1,edgecolor='k',facecolor='none') # draw outer boundary
        ax.add_patch(rect_bdry)
        
        horz_margin = 0.02 # frac to exclude on left/right
        vert_margin = 0.05 # frac to exclude on top/bottom
        xfrac = 0.75       # frac of (layer+sep) given to layer
        Dy_max = 1 - 2*vert_margin # max height of rectangles (as frac)
        Dy_min = 0.05 # min height of rectangles (as frac)
        assert (Dy_min < Dy_max)
        
        ######################
        # Currently only for Sequential family
        ######################
        # draw L rectangles, one for each layer, of equal width
        # height proportional to layer width for each rectangle
        L = params['L'] # number of layers excluding input layer
        dx_full = (1 - 2*horz_margin)/L
        dx = dx_full*xfrac
        xsep = dx_full - dx
        
        min_hw,max_hw = np.min(params['n_layer'][:-1]),np.max(params['n_layer'][:-1]) # min,max widths of hidden layers
        n0 = params['data_dim']
        nlast = params['n_layer'][:-1]
        
        W_min = np.min([n0,min_hw]) # don't incldue nlast (in case nlast=1)
        W_max = np.min([n0,max_hw,nlast])
        W_ratio = W_max/(W_min + 1e-15)
        W = np.concatenate(([n0],params['n_layer']))
        
        if W_ratio >= 10.0:
            # log scaling
            dydw = (Dy_max-Dy_min)/np.log(W_ratio)
            Dy = Dy_max + dydw*np.log(W/W_max)
        elif W_ratio >= 3.0:
            # sqrt scaling
            dydw = (Dy_max-Dy_min)/np.sqrt(W_ratio)
            Dy = Dy_max + dydw*np.sqrt(W/W_max)
        else:
            # linear scaling
            dydW = (Dy_max-Dy_min)/(W_max - W_min + 1e-15)
            Dy = Dy_max + dydW*(W - W_max)

        # IN PROGRESS HERE
        layers = []
        for l in range(L-1):
            # loop over hidden layers
            pass


        # # Loop over data points; create box from errors at each point
        # errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        #               for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

        # # Create patch collection with specified colour/alpha
        # pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
        #                      edgecolor=edgecolor)

        # # Add collection to Axes
        # ax.add_collection(pc)
            
        return 
    ###################
    
#############################################
