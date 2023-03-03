import numpy as np
import os
import sys
import math
import scipy.optimize as optimize
from time import gmtime, strftime
import warnings
warnings.filterwarnings('ignore') # make the notebook nicer
from shutil import copyfile
from collections import namedtuple
import subprocess, threading
import pandas as pd
import pickle
from pickle import Pickler, Unpickler
from collections import OrderedDict, defaultdict
import analyzetool
import analyzetool.gas as gasAnalyze
import analyzetool.liquid as liqAnalyze
from IPython.display import Latex

def save_pickle(dict_,outfn=None):
    if outfn == None:
        my_var_name = [ k for k,v in locals().iteritems() if v == dict_][0]
        outfn = my_var_name
    pickle_out = open(outfn,"wb")
    pickle.dump(dict_, pickle_out)
    pickle_out.close()
    
def load_pickle(filenm):
    pickle_in = open(filenm,"rb")
    example_dict = pickle.load(pickle_in)
    pickle_in.close()
    return example_dict

def dataframe_properties(df): 
    col_names = df.axes[1].to_numpy()
    index_mol = df.axes[0].to_numpy()
    labels = df[col_names[0]].to_numpy()
    return col_names,index_mol,labels 

def main():
    """Usage: python liq_avg.py Water 298.15 liquid.xyz liquid.log 100 analysis.log gas2.log"""
    mol_name = str(sys.argv[1])
    temp = float(sys.argv[2])
    xyzfile = str(sys.argv[3])
    logfile = str(sys.argv[4])
    equil_t = int(sys.argv[5])

    try:
        analyzelog = str(sys.argv[6])
        gaslog = str(sys.argv[7])
    except:
        analyzelog = None
        gaslog = None

    try:
        gaslog = str(sys.argv[7])
    except:
        gaslog = None

    if analyzelog == 'None':
        analyzelog = None

    if logfile == 'None':
        logfile = None

    sim_path = os.getcwd()

    liquid = liqAnalyze.Liquid(sim_path,xyzfile,n_atoms_mol=mol_name,temperature=temp,equil=equil_t,
                 logfile=logfile, analyzelog=analyzelog,gaslog=gaslog)

    np.set_printoptions(threshold=1000,precision=4,linewidth=100)

    
    if gaslog:
        liquid.all_properties(gaslog,analyzelog)

    if os.path.isfile("./diffusion.log"):
        liquid.get_diffusion("diffusion.log")
        diff = liquid.median_diffusion
    else:
        diff = 0.0

    dens = liquid.avgRho
    PEmol = liquid.PEmol

    print("%s @ %.2f K" % (mol_name,temp))
    print("Avg. Density: %5.2f kg/m^3" % (dens*1000))
    print("Avg. PE/mol : %5.2f kcal/mol" % (PEmol))
    print("Heat Capac. : %5.2f cal/mol/K" % (liquid.Cp))
    print("Isot.Compr. : %5.2f 10^-6 bar" % (liquid.kappa))
    print("T.Exp.Coef. : %5.2f 10^-4 1/K" % ((1e4)*liquid.alpha))
    print("Self-diff.  : %5.2f 10^-5 cm^2/s" % (diff))

    if gaslog:
        print("Heat Vapor. : %5.2f Kcal/mol" % (liquid.HV))
        print("Gas Avg.PE. : %5.2f Kcal/mol" % (liquid.gasAvgPE))
    if analyzelog:
        print("Dielectric  : %5.2f " % (liquid.dielectric))

    liquid.get_coordinates('%s/liquid.arc' % sim_path)
    liquid.compute_avg_angle()
    print("Avg. Angle  : %5.2f deg" % (liquid.avg_angle))

    sys.stdout.flush()

if __name__ == "__main__":
    main()
