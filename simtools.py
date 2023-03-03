import numpy as np
import os
import numpy as np
import analyzetool
import analyzetool.gas as gasAnalyze
import analyzetool.liquid as liqAnalyze
import warnings
warnings.filterwarnings('ignore') # make the notebook nicer

KB_J = 1.38064852e-23 #J/K
E0 = 8.854187817620e-12
N_vis = 8.91e-4 #Pa.s
R=1.9872036E-3
NA=6.02214129*(1e23)

KB = 1.38064852E-16 #J/K
statC=4.80320425E-10
charge = 1.602176634E-19
deb_conv = ((1e-20)*(charge/statC))
P_PA = 101325.0

#Water constants
MW = 18.01528 #g/mol

def box_size(dens,N=512):
    density=dens*(1e-27)
    mol_w = MW/NA
    bs = np.cbrt((N*mol_w)/density)
    return bs

def num_molecules(box_size,MW,dens):
    density=dens/(1e24)
    mol_w = MW/NA
    num_molecules = (np.power(box_size,3)*density)/mol_w
    return np.rint(num_molecules)

def calc_density(box_size,n_mol,MW):
    mol_w = MW/NA
    dens = (n_mol*mol_w)/(np.power(box_size,3))
    return dens*(1e24)
    
def calc_box_s(n_mol,MW,dens):
    density=dens/(1e24)
    mol_w = MW/NA
    box_s = np.cbrt((n_mol*mol_w)/density)
    return box_s

### Temperature dependence analysis
def process_liqtemp(base_dir,equil=500,gas_dir=None,exclude=[],angles=False):

    sdirs = next(os.walk(base_dir))[1]
    temps = []
    for fn in sdirs:
        try:
            t = int(fn.split('_')[1])
            temps.append(t)
        except:
            None
            
    temps = sorted(temps)
    
    resdir = f'{base_dir}/results'
    os.system(f'mkdir -p {resdir}')
    
    results = []
    stdres = []
    prop_mol = []
    
    temp_list = np.array([248.15, 258.15, 268.15, 277.15, 288.15, 298.15, 308.15, 318.15,
       328.15, 336.15, 343.15, 353.15, 363.15, 373.15, 383.15, 398.15,
       408.15, 423.15, 243.15, 253.15, 263.15, 273.15, 283.15, 293.15,
       303.15, 313.15])
    
    for tsav in temps:
    #for T in temp_list:
        #tsav = int(T)
        T = tsav + 0.15
        sim_path = f'{base_dir}/sim_{tsav}'
        
        if T in exclude or tsav in exclude:
            continue
        
        if gas_dir is not None:
            gas_path = f"{gas_dir}/sim_{tsav}"
        else:
            gas_path = sim_path
        
        PEmol = 0
        try:
            if os.path.isfile(f'{sim_path}/analysis.log'):
                liquid = liqAnalyze.Liquid(sim_path,'liquid.xyz',3,T,equil,
                                     logfile='liquid.log',analyzelog='analysis.log',gaslog=f'{gas_path}/gas.log')

                liquid.all_properties(f'{gas_path}/gas.log',f'{sim_path}/analysis.log')
                error = False
                PEmol = liquid.PEmol
            else:
                error = True
        except:
            print(f"Something wrong with {tsav} K simulation")
            error = True
        try:
            liquid.get_diffusion(f'{sim_path}/diffusion.log')
        except:
            None
        
        avg_angle = 0
        if angles:
            liquid.get_coordinates(f'{sim_path}/liquid.arc')
            liquid.compute_avg_angle()
            avg_angle = liquid.avg_angle
            
        if not error:
            if np.abs(PEmol) > 0: 
                results.append([T,liquid.avgRho,liquid.HV,0,0,liquid.kappa,liquid.dielectric,
                                liquid.median_diffusion,avg_angle])
                
                #stdres.append([T,liquid.stdRho,liquid.stdHV-liquid.stdGasPE,liquid.stdCp, liquid.stdAlpha,
                stdres.append([T,liquid.stdRho,liquid.stdHV,liquid.stdCp, liquid.stdAlpha,
                               liquid.stdKappa,liquid.stdEps,0,0])
                
                prop_mol.append([T,liquid.PEmol,liquid.avgVol])     
                #print(T,liquid.PEmol,liquid.avgVol)

    stdres = np.array(stdres)
    
    PE_temp = np.array(prop_mol)[:,1]   
    Vol_temp = np.array(prop_mol)[:,2]   
    temp = np.array(prop_mol)[:,0]
    #print(prop_mol)

    aa = np.polyfit(temp,(1000*PE_temp),5)
    pol = np.poly1d(aa)
    ders1 = np.polyder(pol)
    cp = ders1(temp)+(9.0/2.0)*(1e3)*R
    
    aa = np.polyfit(temp,Vol_temp,6)
    pol = np.poly1d(aa)
    ders = np.polyder(pol)
    ap = ders(temp)/(Vol_temp)

    for k, res in enumerate(results):
        results[k][3] = cp[k]
        results[k][4] = ap[k]
        
    np.save(f'{resdir}/prop_vs_T.npy',results)
    np.save(f'{resdir}/std_vs_T.npy',stdres)
    
def process_liqpres(base_dir,equil=500):

    sdirs = next(os.walk(base_dir))[1]
    temps = []
    for fn in sdirs:
        try:
            t = int(fn.split('_')[1])
            temps.append(t)
        except:
            None
            
    temps = sorted(temps)
    
    resdir = f'{base_dir}/results'
    os.system(f'mkdir -p {resdir}')
    
    results = []
    stdres = []
        
    for tsav in temps:
    #for T in temp_list:
        #tsav = int(T)
        T = tsav + 0.15
        sim_path = f'{base_dir}/sim_{tsav}'
                
        PEmol = 0
        try:
            if os.path.isfile(f'{sim_path}/analysis.log'):
                liquid = liqAnalyze.Liquid(sim_path,'liquid.xyz',3,T,equil,
                                     logfile='liquid.log',analyzelog='analysis.log',gaslog=None)

                error = False
                PEmol = liquid.PEmol
            else:
                error = True
        except:
            print(f"Something wrong with {tsav} K simulation")
            error = True
            
        if not error:
            if np.abs(PEmol) > 0: 
                results.append([T,liquid.avgRho,liquid.stdRho])
                
    results = np.array(results)
    
    np.save(f'{resdir}/prop_vs_T.npy',results)
    
