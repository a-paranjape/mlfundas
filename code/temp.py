
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


#############################################
class IllustrateNetwork(Utilities):
    """ Illustrate architecture of an existing network or network ensemble. """
    ###################
    def __init__(self,base_dir,file_stem,ensemble,family='seq',verbose=True,logfile=None):
        """ Load existing network or network ensemble and create simple graphical illustration of its architecture.
            -- base_dir: str, path/to/folder where network (ensemble) was originally created from.
            -- file_stem: str, file stem (relative to base_dir) used for defining location of network (ensemble).
            -- ensemble: bool, whether location is an ensemble (True) or single network (False).
            -- family: str [default 'seq']; one of 'seq' (Sequential), 'biseq' (BiSequential), 'gan' (GAN); needed if ensemble is False.
            -- verbose,logfile: usual I/O control
            Methods:
            -- 
        """
        Utilities.__init__(self)
        self.base_dir = base_dir
        self.ensemble = ensemble
        self.verbose = verbose
        self.logfile = logfile

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
    def illustrate_single(self,net,ax):
        """ Create illustration object of single network. 
            -- net: object of type self.family
            --  ax: matplotlib axis object on which to draw
        """
        bdry = np.array([[0,0],[0,1],
                         [1,0],[1,1]]) # define boundary
        bdry = bdry.T
        ax.plot(bdry[0],bdry[1],c='w') # set boundary
        ax.set_xticks([])              # kill tick marks
        ax.set_yticks([])

        # draw L rectangles, one for each layer, of equal width
        # height proportional to layer width for each rectangle

        return 
    ###################
    
#############################################
