import numpy as np
import os
import sys
import copy
import pickle
from analyzetool import auxfitting, prmedit
from analyzetool.process import save_pickle,load_pickle

def save_termfit(termfit,path,n):
    """Drops a file in the directory of fitting with the name of the energy terms 
       for which parameters are currently being fitted
    """
    listterm = " ".join(termfit)
    listterm += '\n'
    with open(f"{path}/{n}/termfit.txt",'w') as file:
        file.write(listterm)

    print(f"Running {listterm}")
    sys.stdout.flush()
    return listterm

# List of energy components for which parameters can be fitted. 
termfit = ['chgpen','dispersion','repulsion',
                'polarize','chgtrn','multipole']

#### UPDATE THESE PATHS IF NEEDED
# ref_data: directory with all the files and target data used for fitting.
ref_data = "/work/roseane/HIPPO/small_molecules/org_molecules/reference-data"

molinfo = load_pickle(f"{ref_data}/database-info/molinfo_dict.pickle")

def runfit(path,n,elfn,cudad):

    os.chdir(path)
    fitpath = f"{path}/{n}"
    molfit = auxfitting.Auxfit(path,n)

    molfit.datadir = ref_data
    molfit.prepare_directories() # Creates directories for carrying out the fit
    
    molfit.process_prm()         # Reads in the parameter file in the reference folder
    molfit.build_prm_list()      # Create a list of parameters to fit based on termfit
    molfit.make_key()            # Make Tinker software key file, based on given params or
                                ## the ones in the starter parameter file
                                    
    molfit.initpotrms = molfit.get_potfit()  ## Run a potential fit analysis of the current parameters
                                             ## In key file
                                

    ## PRESERVE SOME PARAMETERS FROM ORIGINAL PRMFILE,
    ## Those parameters are never parametrized with this script
    preserve_terms = ['opbend','strbnd','torsion','bndcflux','angcflux']
    files = next(os.walk(fitpath))[2]
    dictfn = np.array([f for f in files if 'newprms' in f])
    modtim = [os.path.getmtime(f"{fitpath}/{f}") for f in dictfn]
    modtim = np.array(modtim)
    inds = np.argsort(modtim)
    if len(dictfn) > 0:
        ndict = f"{fitpath}/{dictfn[inds[-1]]}"

        newdict = load_pickle(ndict)
        print(f"Loading previous prmdict: {dictfn[inds[-1]]}\n")

        for term in preserve_terms:
            newdict[term] = copy.deepcopy(molfit.prmdict[term])

        molfit.prmdict = copy.deepcopy(newdict)

    ## molfit.prmdict is a dictionary of the current parameters.
    ## The initial one as made from the reference parameter file
    ## It gets updated when parameters are changed

    testliq = True                  ## Keyword to test if a simulation run with the current parameters
    molfit.nsteps_test = 10000      ## number of steps in the test run
    molfit.fithv = False            ## when running the test, you can fit enthalpy of vaporization
    fitliq = False                  ## Keywork to perform a full fitting, including condensed phased


    ## List in variable termfit will pass the energy components to run a fit
    ## It will allow creation of a parameter list to fit for that specifically 
    ## energy potential

    ## fit_data() is the function that actually runs an optimization of parameters
    ## It can use two optimization algorithms: genetic, differential evolution
    ## Or least-squares (lstsq), both from scipy. 
    ## the arguments passed are (optimizer(genetic,lstsq),fitliq,testliq,diff_step,wide_range)
    ## wide_range argument sets the rules to make the upper and lower bounds on parameters
    ## if wide_range=True, it uses the full allowed interval for the parameter
    ## if wide_range=False, it creates bounds within 10% of parameter initial value, 
    ## for parameter < 5, or 20% for parameters with larger numbers
    ## diff_step only works with lstq
    
    ## FIT MOLECULAR POLARIZABILITY FIRST
    termfit = ['polarize']
    listterm = " ".join(termfit)
    with open(f"{path}/{n}/termfit.txt",'w') as file:
        file.write(listterm)
    molfit.build_prm_list(termfit)

    res = molfit.fit_data('lstsq', fitliq,testliq, 0.05, False)  
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms.pickle")
    
    print(f"Completed mol. pol. fitting \n")
    sys.stdout.flush()

    ## After finishing parametrizing molecular polarizability,
    ##  change molfit.rungas to allow running gas simulation
    ## to better fit the enthalpy of vaporization, set with the
    ## variable molfit.fithv
    testliq = True
    fitliq = False
    molfit.rungas = True
    molfit.nsteps_gas = 500000
    molfit.fithv = True
    os.system(f"touch {path}/{n}/FIT_RUNNING")


    ## The next block looks over the reference directory to see 
    ## what types of quantum calculation data is available to fit
    ## the force field energy to that SAPT component or total energy
    if os.path.isfile(f"{ref_data}/qm-calc/{n}/sapt-res-water+mol.npy"):
        molfit.prepare_opt_sapt_dimers()
    if os.path.isdir(f"{ref_data}/qm-calc/{n}/sapt_dimers"):
        molfit.prepare_sapt_dimers()
    if os.path.isdir(f"{ref_data}/qm-calc/{n}/clusters"):
        molfit.prepare_cluster()
    if os.path.isdir(f"{ref_data}/qm-calc/{n}/ccsdt_dimers"):
        molfit.prepare_ccsdt_dimers()
    
    ######
    ## Function calls to fit energy terms are done bellow
    ## The optimizer can take one or many energy terms at a time.

    ## Induction fitting through fit of charge transfer parameters
    termfit = ['chgtrn']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq,testliq,0.05,False)

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms8.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    ######
    ## Electrostatics fitting through fit of charge penetration parameters
    termfit = ['chgpen']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq,testliq,0.05,False)

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms8.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()


    molfit.nsteps_test = 25000
    molfit.nsteps_gas = 250000
    molfit.rungas = True
    molfit.fithv = True

    ## Repulsion fitting through fit of repulsion parameters
    termfit = ['repulsion']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq,testliq,0.05,False)

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms8.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    ## Dispersion fitting through fit of dispersion parameters
    termfit = ['dispersion']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq,testliq)

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms8.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    ## The next blocks fit more than one energy term at a time
    #Induction
    termfit = ['repulsion','chgtrn']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq,testliq,0.05,False)

    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms9.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    termfit = ['dispersion','repulsion','chgtrn']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq, testliq,0.05,False)
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms9.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    ## The next blocks will turn on fitliq to allow the fit to use
    ## experimental liquid data and run simulation for fitting of the
    ## parameters to reproduce experimental properties
    testliq = False
    fitliq = True
    molfit.fithv = False

    molfit.gasdcd = f"{path}/{n}/ref_liquid/gas.dcd"
    ## molinfo has the experimental information per molecule
    info = molinfo[n]

    ## Use a continuation file .dyn in the liquid fitting
    ## this allows that the box of the simulation is pre-equilibrated
    if os.path.isfile(f"{path}/{n}/ref_liquid/liquid.dcd") and not os.path.isfile(f"{path}/{n}/ref_liquid/liquid.err"):
        molfit.useliqdyn = True

    ## Change some of the references for two specific cases, not an
    ## essential part of the fitting, but allows to change the reference
    ## case by case
    if n == 6:     # Methanoic acid
        molfit.liquidref[0][2] = 2*molfit.liquidref[0][2] + 6.76792
    if n == 147:   # Acetic acid
        molfit.liquidref[0][2] = 2*molfit.liquidref[0][2] + 5.53816
    
    ## info[0] is the temperature of the simulatin. This code block will
    ## increase the lenght of the simulation if the temperature is lower than
    ## a threshold because lower temperature simulations converges slower

    if info[0] < 275:
        molfit.nsteps = 1500000
        molfit.equil = 1500
    elif info[0] < 250:
        molfit.nsteps = 2000000
        molfit.equil = 2000
    else:
        molfit.nsteps = 500000
        molfit.equil = 500
        molfit.useliqdyn = False
    
    molfit.rungas = True
    molfit.nsteps_gas = 5000000
    molfit.gasdcd = f"{path}/{n}/ref_liquid/gas.dcd"

    ## Use dispersion, repulsion and chgtrn parameters to fit to 
    ## experimental data targets
    termfit = ['dispersion','repulsion','chgtrn']
    listterm = save_termfit(termfit,path,n)
    molfit.build_prm_list(termfit)
    res = molfit.fit_data('lstsq', fitliq, testliq)
    molfit.prmdict = molfit.prmlist_to_dict(res.x)
    # Save dictionary of newprms
    save_pickle(molfit.prmdict,f"{path}/{n}/newprms10.pickle")

    print(f"Completed {listterm}")
    sys.stdout.flush()

    ## Save results
    if not os.path.isdir(f"{path}/{n}/fit_results"):
        os.system(f"mkdir -p {path}/{n}/fit_results")

    resdir = f"{path}/{n}/fit_results"
    fname = f"{resdir}/{n}-latest.prm"
    num = 1
    files = next(os.walk(resdir))[2]
    files = [a for a in files if f'{n}' in a and 'prm' in a]
    if len(files) == 0:
        None
    
    elif len(files) == 1:
        if f"{n}-latest.prm" == files[0]:
            os.system(f"cp {fname} {resdir}/{n}-{num:02d}.prm")
        
    else:
        files = sorted([f for f in files if 'latest' not in f])
        num = int(files[-1][:-4].split('-')[-1])
        num += 1

        os.system(f"cp {fname} {resdir}/{n}-{num:02d}.prm")

    origprm = f"{ref_data}/prmfiles/{n}.prm"
    prm1 = prmedit.process_prm(origprm)
    mfacts = prm1['multipole_factors']
    prmedit.write_prm(molfit.prmdict,fname,mfacts)

    print(f"Finished!")
    sys.stdout.flush()

    os.system(f"rm {path}/{n}/FIT_RUNNING")

def main():
    n = int(sys.argv[1])

    try:
        elfn = int(sys.argv[2])
    except:
        elfn = 0
    
    try:
        cudad = int(sys.argv[3])
    except:
        cudad = -1

    path = "/user/roseane/HIPPO/small_molecules/org_molecules/fitting-2"
    # path = os.getcwd()
    ## You can set this path to be your current directory 
    ## this path is where the fitting directories are going to be
    ## Created for every molecules. Molecules are always referred to
    ## by numerical ID

    runfit(path,n,elfn,cudad)

    ## Elfn and cudad are variables specific to a local cluster to run 
    ## simulations

if __name__ == "__main__":
    main()
