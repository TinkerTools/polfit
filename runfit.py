import numpy as np
import os
import sys
import math
import scipy.optimize as optimize
from time import gmtime, strftime
import warnings
warnings.filterwarnings('ignore') # make the notebook nicer
import pickle

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

termfit = ['chgpen','dispersion','repulsion',
                'polarize','bndcflux','angcflux',
                'chgtrn','multipole']

def main():
    n = int(sys.argv[1])

    bspath = "/work/roseane/HIPPO/small_molecules/org_molecules/fitting"
    path = os.getcwd()

    if os.path.isfile(f"{path}/auxfunctions.py"):
        sys.path.append(path)
        import auxfunctions
    elif os.path.isfile(f"{bspath}/auxfunctions.py"):
        sys.path.append(bspath)
        import auxfunctions
    else:
        return

    os.chdir(path)

    molfit = auxfunctions.Auxfit(path,n)

    molfit.prepare_directories()
    # molfit.prepare_opt_sapt_dimers()
    molfit.prepare_ccsdt_dimers()
    # molfit.prepare_cluster()
    # molfit.prepare_sapt_dimers()
    molfit.process_prm()
    molfit.build_prm_list()
    molfit.make_key()
    molfit.initpotrms = molfit.get_potfit()

    
    termfit = ['chgpen']
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('genetic')

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")

    print(f"Completed chgpen \n")
    sys.stdout.flush()

    #Disp
    termfit = ['dispersion']
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq')

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")
    print(f"Completed dispersion \n")
    sys.stdout.flush()

    # # #Repulsion
    termfit = ['repulsion']
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq')
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")
    print(f"Completed repulsion \n")
    sys.stdout.flush()

    ######
    #Induction
    termfit = ['polarize']
    listterm = " ".join(termfit)
    with open(f"{path}/{n}/termfit.txt",'w') as file:
        file.write(listterm)
    molfit.build_prm_list(termfit)

    molfit.progfile = f'{path}/{n}/progress.pickle'
    res = molfit.fit_data('lstsq')
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")

    print(f"Completed polarize \n")

    termfit = ['chgtrn']
    listterm = " ".join(termfit)
    with open(f"{path}/{n}/termfit.txt",'w') as file:
        file.write(listterm)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq')
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")

    print(f"Completed Induction \n")

    print(f"Finished!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()