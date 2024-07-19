import numpy as np
import os, sys
import mdtraj as md
from . import gas
from .process import mw_elements, type_map, count_atoms
kB = 0.008314472471220214
atm_unit = 0.061019351687175
bar_unit = 0.060221417930000
# prefactor = 30.348705333964077

KB_J = 1.38064852e-23 #J/K
E0 = 8.854187817620e-12
DEB = 3.33564095198152e-30
R=1.9872036E-3

### dielectric calculation prefactor
# prefactor = (1/3)*(DEB**2)/(KB_J*E0*1e-30)
prefactor = 30.3392945e3


tinkerpath = '/user/roseane/tinker'
data_dir = '/user/roseane/HIPPO/reference_data'

n_atoms_per_mol = {'Methanol': 6,
                   'MethylChloride': 5,
                   'Ethene': 6,
                   'Imidazolium': 10,
                   'Uracil': 12,
                   'PhosphoricAcid': 8,
                   'MethyleneFluoride': 5,
                   'MethylAmine': 7,
                   'MethylBromide': 5,
                   'AceticAcid': 8,
                   'MethyleneChloride': 5,
                   'MethylAmmonium': 8,
                   'Acetamide': 9,
                   'MethylSulfide': 6,
                   'Phenol': 13,
                   'Neopentane': 17,
                   'ChloroBenzene': 12,
                   'Acetate': 7,
                   'FluoroBenzene': 12,
                   'DihydrogenPhosphate': 7,
                   'MethyleneBromide': 5,
                   'BromoBenzene': 12,
                   'Indole': 16,
                   'MethylFluoride': 5,
                   'DimethylSulfide': 9,
                   'Pyrrolidine': 14,
                   'MethylAcetamide': 12,
                   'Pentane': 17,
                   'Pyridine': 11,
                   'Water': 3,
                   'HydrogenPhosphate': 6,
                   'DimethylSulfoxide': 10,
                   'Benzene': 12,
                   'Guanidinium': 10,
                   'Imidazole': 9,
                   'Cyclopentane': 15}

def bzavg(obs,boltz):
    """ Get the Boltzmann average of an observable. """
    if obs.ndim == 2:
        if obs.shape[0] == len(boltz) and obs.shape[1] == len(boltz):
            raise Exception('Error - both dimensions have length equal to number of snapshots, now confused!')
        elif obs.shape[0] == len(boltz):
            return np.sum(obs*boltz.reshape(-1,1),axis=0)/np.sum(boltz)
        elif obs.shape[1] == len(boltz):
            return np.sum(obs*boltz,axis=1)/np.sum(boltz)
        else:
            raise Exception('The dimensions are wrong!')
    elif obs.ndim == 1:
        return np.dot(obs,boltz)/sum(boltz)
    else:
        raise Exception('The number of dimensions can only be 1 or 2!')

### FROM FORCEBALANCE 
#===========================================#
#| John's statisticalInefficiency function |#
#===========================================#
def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3, warn=True):

    """
    Compute the (cross) statistical inefficiency of (two) timeseries.

    Notes
      The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
      The fast method described in Ref [1] is used to compute g.

    References
      [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
      histogram analysis method for the analysis of simulated and parallel tempering simulations.
      JCTC 3(1):26-41, 2007.

    Examples

    Compute statistical inefficiency of timeseries data with known correlation time.

    >>> import timeseries
    >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)

    @param[in] A_n (required, numpy array) - A_n[n] is nth value of
    timeseries A.  Length is deduced from vector.

    @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
    timeseries B.  Length is deduced from vector.  If supplied, the
    cross-correlation of timeseries A and B will be estimated instead of
    the autocorrelation of timeseries A.

    @param[in] fast (optional, boolean) - if True, will use faster (but
    less accurate) method to estimate correlation time, described in
    Ref. [1] (default: False)

    @param[in] mintime (optional, int) - minimum amount of correlation
    function to compute (default: 3) The algorithm terminates after
    computing the correlation time out to mintime when the correlation
    function furst goes negative.  Note that this time may need to be
    increased if there is a strong initial negative peak in the
    correlation function.

    @return g The estimated statistical inefficiency (equal to 1 + 2
    tau, where tau is the correlation time).  We enforce g >= 1.0.

    """
    # Create numpy copies of input arguments.
    A_n = np.array(A_n)
    if B_n is not None:
        B_n = np.array(B_n)
    else:
        B_n = np.array(A_n)
    # Get the length of the timeseries.
    N = A_n.shape[0]
    # Be sure A_n and B_n have the same dimensions.
    if A_n.shape != B_n.shape:
        print('A_n and B_n must have same dimensions.\n')
    
    # Initialize statistical inefficiency estimate with uncorrelated value.
    g = 1.0
    # Compute mean of each timeseries.
    mu_A = A_n.mean()
    mu_B = B_n.mean()
    # Make temporary copies of fluctuation from mean.
    dA_n = A_n.astype(np.float64) - mu_A
    dB_n = B_n.astype(np.float64) - mu_B
    # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
    # Trap the case where this covariance is zero, and we cannot proceed.
    if sigma2_AB == 0:
        print('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency\n')
    # Accumulate the integrated correlation time by computing the normalized correlation time at
    # increasing values of t.  Stop accumulating if the correlation function goes negative, since
    # this is unlikely to occur unless the correlation function has decayed to the point where it
    # is dominated by noise and indistinguishable from zero.
    t = 1
    increment = 1
    while t < N-1:
        # compute normalized fluctuation correlation function at time t
        C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
        # Terminate if the correlation function has crossed zero and we've computed the correlation
        # function at least out to 'mintime'.
        if (C <= 0.0) and (t > mintime):
            break
        # Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
        # Increment t and the amount by which we increment t.
        t += increment
        # Increase the interval if "fast mode" is on.
        if fast: increment += 1
    # g must be at least unity
    if g < 1.0: g = 1.0
    # Return the computed statistical inefficiency.
    return g

def mean_stderr(ts):
    """Return mean and standard deviation of a time series ts."""
    error = False
    try:
        tsmean = np.mean(ts)
        ts_std = np.std(ts)*np.sqrt(statisticalInefficiency(ts, warn=False)/len(ts))
        return tsmean,ts_std,error
    except:
        error = True
        return 0,0,error

def convert_float(string):
    a = True
    try:
        a = float(string)
        err = False
    except:
        err = True
        a = -100
    return a, err

def compute_angle(a,b,c):
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def compute_bond(a,b):
    a = np.asarray(a)
    b = np.asarray(b)

    ba = b - a

    bond = np.sqrt((ba**2).sum())

    return bond

def has_numbers(inputString):
    try:
        float(inputString)
        return True
    except:
        return False
    # return any(char.isdigit() for char in inputString)

numboots = 1000

class Liquid(object):
    def __init__(self, sim_path,xyzfile='liquid.xyz',n_atoms_mol=None,temperature=298.15,equil=200,
                 logfile=None, analyzelog=None, sysinfo=None, gaslog=None,molpol=None):

        """syntax : Liquid(simulation_path, xyz_file_name,
        number of atoms per molecule or molecule name to search database,
        temp. of simulation,
        number of steps to consider as equilibration,
        name of log file of dynamics, name of analyze log file (optional)"""

        R=1.9872036e-3
        self.kT = R * temperature
        self.equil = equil
        self.path = sim_path
        self.T = temperature
        self.sysinfo = sysinfo
        self.N_vis = 8.91e-4 #Pa.s water

        if '.xyz' not in xyzfile:
            xyzfile += '.xyz'
            self.xyzfile = xyzfile

        if isinstance(n_atoms_mol, int) or isinstance(n_atoms_mol, float):
            Nmonomer = int(n_atoms_mol)
            #self.Natoms = int(n_atoms_mol)
        elif isinstance(n_atoms_mol, str):
            try:
                Nmonomer = n_atoms_per_mol[n_atoms_mol]
                #self.Natoms = n_atoms_per_mol[n_atoms_mol]
            except:
                if os.path.isfile(n_atoms_mol):
                    Nmonomer = count_atoms(n_atoms_mol)
                elif os.path.isfile(f"{sim_path}/{n_atoms_mol}"):
                    Nmonomer = count_atoms(f"{sim_path}/{n_atoms_mol}")
                # raise Exception('Cannot obtain number of atoms in molecule, please specify')

        self.xyzfile = xyzfile

        xyz_fn = os.path.join(sim_path,xyzfile)
        f = open(xyz_fn)
        lines = f.readlines()
        f.close()

        test = lines[1].split()
        if has_numbers(test[1]):
            n = 2
        else:
            n = 1

        elements = [a.split()[1] for a in lines[n:]]
        total_mass = 0
        for a in elements:
            try:
                m = float(mw_elements[a])
            except:
                e1 = type_map[a]
                m = float(mw_elements[e1])
            total_mass += m
 

        self.mass = total_mass
        self.Nmol = int(len(elements)/Nmonomer)
        self.Nmonomer = Nmonomer
        self.Natoms = int(len(elements))

        R=1.9872036e-3
        T = temperature
        kT = R * T
        mBeta = -1.0 / kT
        Beta = 1.0 / kT

        self.avgPE = 0
        self.avgKE = 0
        self.gasAvgPE = 0

        self.KE = []
        self.PE = []
        self.gasPE = []

        edyn = []
        kdyn = []
        dens = []
        vol = []

        self.H = []
        self.avgPE = 0
        self.avgKE = 0
        
        self.PEmol = 0
        self.avgVol = 0
        self.avgRho = 0
        self.median_diffusion = 0
        self.alpha = 0
        self.kappa = 0
        self.Cp = 0

        self.Vol = []
        self.Rhos = []
        self.lattice = []
        self.error = False
        self.gasfn = None
        self.nframes = 0
        error = []
        self.coords = np.array([])

        if sysinfo is not None:
            self.system_info()

        if logfile is not None:
            liqlog = os.path.join(sim_path,logfile)

            if os.path.isfile(liqlog):
                f = open(liqlog,'r')
                dt = f.readlines()
                f.close()

                for line in dt:
                    s = line.split()
                    if 'Current Potential' in line:
                        val, err = convert_float(s[2])
                        edyn.append(val)
                        error.append(err)
                    if 'Current Kinetic' in line:
                        val, err = convert_float(s[2])
                        kdyn.append(val)
                    if 'Density' in line:
                        val, err = convert_float(s[1])
                        dens.append(val)
                    if 'Lattice Lengths' in line:
                        val1, err = convert_float(s[2])
                        val2, err = convert_float(s[3])
                        val3, err = convert_float(s[4])
                        vol.append(val1*val2*val3)

                if any(error):
                    self.error = True
                
                eq = int(equil)
                self.PE = np.array(edyn)
                self.KE = np.array(kdyn)

                nframes = self.PE.shape[0]
                self.nframes = nframes
                if nframes < eq:
                    if nframes > 1:
                        eq = int(nframes/2)
                        self.equil = eq
                    else:
                        self.error = True


                if not self.error:
                    frm_nm = min(self.PE.shape[0],self.KE.shape[0])
                    self.H = self.PE[:frm_nm]+self.KE[:frm_nm]
                    self.avgPE = self.PE[eq:].mean()
                    self.avgKE = self.KE[eq:].mean()
                    
                    self.PEmol = self.avgPE/self.Nmol

                    if len(self.Vol) == 0:
                        vol = np.array(vol)
                    else:
                        vol = self.Vol

                    error = self.error
                    if len(vol) > len(self.Vol):
                        conv = 1.6605387831627252
                        Rhos = conv * self.mass / vol
                    
                        self.avgVol, self.stdVol, error = mean_stderr(vol[eq:])
                        self.avgRho, self.stdRho, error = mean_stderr(Rhos[eq:])

                        if not error:
                            self.Vol = np.copy(vol)
                            self.Rhos = np.copy(Rhos)

                    if not error:
                        self.HV = 0
                        L = self.H[eq:].shape[0]
                        
                        self.alpha = self.calc_alpha()
                        self.Cp = self.calc_cp()
                        self.kappa = self.calc_kappa()
                        
                        self.dielectric = 0
                        self.Dips = []
                    else:
                        self.error = True

            else:
                self.error = True

            if analyzelog is not None and not self.error:
                anal_fn = os.path.join(sim_path,analyzelog)
            
                if os.path.isfile(anal_fn):
                    f = open(anal_fn,'r')
                    dt = f.readlines()
                    f.close()

                    self.analyzelog = anal_fn

                    log = [a.strip('\n') for a in dt]
                    dip = []
                    for line in log:
                        s = line.split()
                        if 'Dipole X,Y,Z-Components' in line:
                            dip.append([float(s[i]) for i in range(-3,0)])

                    if len(dip) == len(self.PE):
                        self.Dips = np.array(dip)    
                        if molpol == None:
                            epf_inf = 1
                        else:
                            vmol = self.avgVol/self.Nmol 
                            epf_inf = (-np.pi*8*molpol - 3*vmol)/(np.pi*4*molpol - 3*vmol)
                        self.epf_inf = epf_inf                    
                        self.dielectric = self.calc_eps0()
                        
                    else:
                        self.dielectric = 0
                        self.Dips = []
                else:
                    self.analyzelog = None

        if logfile is None and analyzelog is not None:
            self.all_properties(gaslog,analyzelog,molpol)

        elif gaslog is not None and not self.error:
            if os.path.isfile(os.path.join(self.path,gaslog)):
                gas_fn = os.path.join(self.path,gaslog)
            else:
                gas_fn = gaslog
            self.gasfn = gas_fn
            gasdata = gas.GasLog(gas_fn)
            self.gasAvgPE = gasdata.avgPE
            self.gasPE = gasdata.PE
            self.gasAvgKE = gasdata.avgKE
            self.gasKE = gasdata.KE
            self.stdGasPE = gasdata.stdPE
            try:
                self.gasAvgH = gasdata.avgH
                self.gasH = gasdata.H
            except:
                self.gasAvgH = gasdata.avgPE
                self.gasH = gasdata.PE

            self.Hmol = (self.avgKE+self.avgPE)/self.Nmol

            self.HV2 = self.gasAvgH - (self.Hmol) + kT

            self.HV = self.gasAvgPE - (self.PEmol) + kT

        
    def calc_alpha(self,h_ = [],v_ = [],b = None):
        kT = self.kT
        T = self.T
        eq = self.equil

        if len(h_) == 0:
            h_ = self.H[eq:]
        if len(v_) == 0:
            v_ = self.Vol[eq:]

        L = v_.shape[0]
        L2 = h_.shape[0]

        if L != L2:
            L = min(L,L2)
            h_ = h_[:L]
            v_ = v_[:L]
        if b is None: b = np.ones(L,dtype=float)

        return 1/(kT*T) * (bzavg(h_*v_,b)-bzavg(h_,b)*bzavg(v_,b))/bzavg(v_,b)

    def calc_kappa(self,v_ = []):
        KB_J = 1.38064852E-23
        eq = self.equil
        T = self.T

        if len(v_) == 0:
            V0 = (1e-30)*self.Vol[eq:]
        else:
            V0 = (1e-30)*v_
        
        volume_squared = V0*V0
        avg_volume = V0.mean()
        volume_fluct = (volume_squared.mean()-(avg_volume*avg_volume))

        return (1e11)*(volume_fluct/(KB_J*T*avg_volume))

    def calc_cp(self,h_=[],b=None):
        Nmol = self.Nmol
        eq = self.equil
        kT = self.kT
        T = self.T

        if len(h_) == 0:
            h_ = self.H[eq:]
        
        T = self.T
        kT = self.kT      
        L = h_.shape[0]
        if b is None: b = np.ones(L,dtype=float)
        Cp_  = 1/(Nmol*kT*T) * (bzavg(h_**2,b) - bzavg(h_,b)**2)
        Cp_ *= 1000
        return Cp_

    def calc_eps0(self,d_=[],v_=[],b=None):
        eq = self.equil
        epf_inf = self.epf_inf

        if len(v_) == 0:
            d_ = self.Dips
            v_ = self.Vol

            ### 
            L1 = d_.shape[0]
            L2 = v_.shape[0]

            if L1 > L2:
                extra = L1-L2
                d_ = d_[extra+eq:]
                v_ = v_[eq:]
                L = v_.shape[0]
            elif L1 < L2:
                extra = L2-L1
                d_ = d_[eq:]
                v_ = v_[eq:-extra]
                L = d_.shape[0]
            else:
                d_ = d_[eq:]
                v_ = v_[eq:]
                L = v_.shape[0]
        else:
            L = d_.shape[0]

        T = self.T

        if b is None: b = np.ones(L,dtype=float)
        dx = d_[:,0]
        dy = d_[:,1]
        dz = d_[:,2]
        D2  = bzavg(dx**2,b)-bzavg(dx,b)**2
        D2 += bzavg(dy**2,b)-bzavg(dy,b)**2
        D2 += bzavg(dz**2,b)-bzavg(dz,b)**2
        return epf_inf + prefactor*D2/bzavg(v_,b)/T


    def system_info(self):
        if os.path.isfile(self.sysinfo):
            filenm = self.sysinfo
        elif os.path.isfile(f"{self.path}/{self.sysinfo}"):
            filenm = f"{self.path}/{self.sysinfo}"
        else:
            return
        
        f = open(filenm,'r')
        dt = f.readlines()
        f.close()

        log = [a.strip('\n') for a in dt]

        lattice = []
        dens = []
        vols = []
        mass = 0.0

        for ln, line in enumerate(log):
            s = line.split()
            if 'Total System Mass' in line:
                mass = float(s[-1])
            if 'System Density' in line:
                dens.append(float(s[-1]))
            if 'a-Axis Length' in line:
                lvec = []
                for vline in log[ln:ln+6]:
                    sv = vline.split()
                    lvec.append(float(sv[-1]))
                lattice.append(lvec)
            if 'Cell Volume' in line:
                vols.append(float(s[-1]))

        
        if mass > 0:
            self.mass = mass

        if len(vols) > 0:
            self.Vol = np.array(vols)
        if len(dens) > 0:
            self.Rhos = np.array(dens)
        if len(lattice) > 0:
            self.lattice = np.array(lattice)

        if len(self.Vol) > 0:
            conv = 1.6605387831627252
            Rhos = conv * self.mass / self.Vol
            self.Rhos = np.copy(Rhos)
            eq = self.equil
            self.avgVol, self.stdVol, self.error = mean_stderr(self.Vol[eq:])
            self.avgRho, self.stdRho, self.error = mean_stderr(Rhos[eq:])
    
    def all_properties(self,gaslog=None,analyzelog=None,molpol=None):
        """syntax : to get all properties you must specify a path for a gas 
        log file. then, specify a liquid analyze log file.
        all_properties(gaslog,analyzelog)"""
        
        if analyzelog is None:
            try:
                anal_fn = self.analyzelog
            except:
                print('Must specify analyze log file')
                return
        else:
            if not os.path.isfile(analyzelog):
                anal_fn = os.path.join(self.path,analyzelog)
            else: 
                anal_fn = analyzelog

        if gaslog is None:
            try:
                gaslog = self.gasfn
            except:
                None
        else:
            if os.path.isfile(f"{self.path}/{gaslog}"):
                gaslog = f"{self.path}/{gaslog}"
            elif not os.path.isfile(gaslog):
                gaslog = None

            self.gasfn = gaslog
        
        if os.path.isfile(anal_fn):
            f = open(anal_fn,'r')
            dt = f.readlines()
            f.close()

            log = [a.strip('\n') for a in dt]
        else:
            raise Exception('The file %s does not exist' % anal_fn)

        Volumes = []
        if self.sysinfo != None:
            self.system_info()
            Volumes = self.Vol
        
        eanl = []
        dip = []
        vols = []
        mass = 0.0

        for ln, line in enumerate(log):
            s = line.split()
            if 'Total System Mass' in line:
                mass = float(s[-1])
            if 'Total Potential Energy : ' in line:
                eanl.append(float(s[4]))
            if 'Dipole X,Y,Z-Components :' in line:
                dip.append([float(s[i]) for i in range(-3,0)])
            if 'Cell Volume' in line:
                vols.append(float(s[-1]))

        if len(Volumes) == 0 and len(vols) > 0:
            Volumes = np.array(vols)
        
        if mass > 0:
            self.mass = mass
            
        Potentials = np.array(eanl)
        
        T = self.T
        R=1.9872036E-3
        kT = R * T

        self.RT = kT

        self.PE = np.array(Potentials)
        eq = self.equil

        nframes = self.PE.shape[0]
        self.nframes = nframes
        if nframes < eq:
            if nframes != 0:
                eq = int(nframes/2)
                self.equil = eq
            else:
                self.error = True
                return

        self.avgPE, self.stdPE, self.error = mean_stderr(self.PE[eq:])

        if self.error :
            return
        
        self.PEmol = self.avgPE/self.Nmol

        if Volumes.shape[0] > len(self.Vol):
            conv = 1.6605387831627252
            Rhos = conv * self.mass / Volumes
            self.avgVol, self.stdVol, self.error = mean_stderr(Volumes[eq:])
            self.avgRho, self.stdRho, self.error = mean_stderr(Rhos[eq:])
            self.Vol = np.copy(Volumes)
            self.Rhos = np.copy(Rhos)
        
        if self.error:
            return

        if len(dip) != 0:
            self.Dips = np.array(dip)
            ### compute eps_inf
            if molpol is None or molpol <= 0:
                epf_inf = 1
            else:
                vmol = self.avgVol/self.Nmol 
                epf_inf = (-np.pi*8*molpol - 3*vmol)/(np.pi*4*molpol - 3*vmol)
            self.epf_inf = epf_inf
            self.dielectric = self.calc_eps0()
        else:
            self.dielectric = 0

        if len(self.H) == 0:
            self.H = self.PE

        self.alpha = self.calc_alpha()
        self.alpha2 = self.calc_alpha(self.PE[eq:],self.Vol[eq:])

        self.Cp = self.calc_cp()
        self.kappa = self.calc_kappa()

        if gaslog is not None:
            if os.path.isfile(os.path.join(self.path,gaslog)):
                gas_fn = os.path.join(self.path,gaslog)
            else:
                gas_fn = gaslog
            gasdata = gas.GasLog(gas_fn)
            self.gasAvgPE = gasdata.avgPE
            self.gasPE = gasdata.PE
            self.gasAvgKE = gasdata.avgKE
            self.gasKE = gasdata.KE
            self.stdGasPE = gasdata.stdPE
            try:
                self.gasAvgH = gasdata.avgH
                self.gasH = gasdata.H
            except:
                self.gasAvgH = gasdata.avgPE
                self.gasH = gasdata.PE
        
        self.Hmol = self.H/self.Nmol
        self.HV2 = self.gasAvgH - (self.Hmol) + kT
        self.HV = self.gasAvgPE - (self.PEmol) + kT
        
        # This is how I calculated the prefactor for the dielectric constant.
        # eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
        # epsunit = 1.0*(debye**2) / nanometer**3 / BOLTZMANN_CONSTANT_kB / kelvin
        # prefactor = epsunit/eps0/3

        #### Standard deviation
        if self.gasAvgPE > 0:
            self.stdHV = np.sqrt(gasdata.stdPE**2 + (self.stdPE/self.Nmol)**2)

        
        L = self.Rhos[eq:].shape[0]
        # N = 10000
        N = int(L/3)
        Rhoboot = []
        for i in range(numboots):
           boot = np.random.randint(L,size=N)
           Rhoboot.append(self.Rhos[eq:][boot].mean())
        Rhoboot = np.array(Rhoboot)
        Rho_err = np.std(Rhoboot)
        self.stdRho = Rho_err * np.sqrt(statisticalInefficiency(self.Rhos[eq:]))

        L = self.Vol[eq:].shape[0]
        N = int(L/3)
        Alphaboot = []
        for i in range(numboots):
            boot = np.random.randint(L,size=N)
            Alphaboot.append(self.calc_alpha(self.H[eq:][boot], self.Vol[eq:][boot]))
        Alphaboot = np.array(Alphaboot)
        self.stdAlpha = np.std(Alphaboot) * max([np.sqrt(statisticalInefficiency(self.Vol[eq:])),
                                                 np.sqrt(statisticalInefficiency(self.H[eq:]))])
        
        Kappaboot = []
        for i in range(numboots):
            boot = np.random.randint(L,size=N)
            Kappaboot.append(self.calc_kappa(self.Vol[eq:][boot]))
        Kappaboot = np.array(Kappaboot)
        self.stdKappa = np.std(Kappaboot) * np.sqrt(statisticalInefficiency(self.Vol[eq:]))

        Cpboot = []
        for i in range(numboots):
            boot = np.random.randint(L,size=N)
            Cpboot.append(self.calc_cp(self.H[eq:][boot]))
        Cpboot = np.array(Cpboot)
        self.stdCp = np.std(Cpboot) * np.sqrt(statisticalInefficiency(self.H[eq:]))

        
        if len(dip) != 0:
            L1 = self.Dips[eq:].shape[0]
            L2 = self.Vol[eq:].shape[0]

            L = np.min((L1,L2))
            N = int(L/3)
            Eps0boot = []
            for i in range(numboots):
                boot = np.random.randint(L,size=N)
                
                if L1 > L2:
                    extra = L1-L2
                    Eps0boot.append(self.calc_eps0(self.Dips[extra+eq:][boot],self.Vol[eq:][boot]))
                elif L1 < L2:
                    extra = L2-L1
                    Eps0boot.append(self.calc_eps0(self.Dips[eq:][boot],self.Vol[eq:-extra][boot]))
                else:
                    Eps0boot.append(self.calc_eps0(self.Dips[eq:][boot],self.Vol[eq:][boot]))
            Eps0boot = np.array(Eps0boot)
            self.stdEps = np.std(Eps0boot)*np.sqrt(np.mean([statisticalInefficiency(self.Dips[:,0][eq:]),
                                                            statisticalInefficiency(self.Dips[:,1][eq:]),
                                                            statisticalInefficiency(self.Dips[:,2][eq:])]))

    def get_dielectric(self,analyzelog,molpol=None):

        if os.path.isfile(os.path.join(self.path,analyzelog)):
            anal_fn = os.path.join(self.path,analyzelog)
        else:
            anal_fn = analyzelog
        
        if os.path.isfile(anal_fn):
            f = open(anal_fn,'r')
            dt = f.readlines()
            f.close()

            log = [a.strip('\n') for a in dt]
        else:
            raise Exception('The file %s does not exist' % anal_fn)

        dip = []
        for ln, line in enumerate(log):
            s = line.split()
            if 'Dipole X,Y,Z-Components :' in line:
                dip.append([float(s[i]) for i in range(-3,0)])

        if len(dip) == 0:
            print("No dipole moment data, dielectric will not be computed")
            self.Dips = []
            self.dielectric = 0
        else:
            self.Dips = np.array(dip)
            ### compute eps_inf
            if molpol == None:
                epf_inf = 1
            else:
                vmol = self.avgVol/self.Nmol 
                epf_inf = (-np.pi*8*molpol - 3*vmol)/(np.pi*4*molpol - 3*vmol)
            self.epf_inf = epf_inf

            self.dielectric = self.calc_eps0()
            

    def get_diffusion(self,analyzelog):

        if os.path.isfile(os.path.join(self.path,analyzelog)):
            anal_fn = os.path.join(self.path,analyzelog)
        else:
            anal_fn = analyzelog
        
        if os.path.isfile(anal_fn):
            f = open(anal_fn,'r')
            dt = f.readlines()
            f.close()

            log = [a.strip('\n') for a in dt]
        else:
            raise Exception('The file %s does not exist' % anal_fn)


        for k,line in enumerate(log):
            tt = line.split()
            a = len(tt)
           
            if a > 0:
                try:
                    b = float(tt[0])
                    break
                except:
                    continue
        
        g_data = log[k:]
        g_data = [a.split() for a in g_data]
        g_data = np.array(g_data,dtype=float)

        T = self.T
        def diff_correction(box_size,T):
            const = (2.837297)/(6*np.pi)
            corr = (1e4)*const*KB_J*T/(self.N_vis*box_size*(1.0e-10))
            return corr

        box_size = np.cbrt(self.avgVol)
        diff_correction = diff_correction(box_size,self.T)
        
        self.diffusion = g_data[:,5]
        self.median_diffusion = np.median(self.diffusion)
        self.diffcorr = (1e5)*diff_correction

    def get_g_r(self,analyzelog):
        if os.path.isfile(os.path.join(self.path,analyzelog)):
            anal_fn = os.path.join(self.path,analyzelog)
        else:
            anal_fn = analyzelog
        
        if os.path.isfile(anal_fn):
            f = open(anal_fn,'r')
            dt = f.readlines()
            f.close()

            log = [a.strip('\n') for a in dt]
        else:
            raise Exception('The file %s does not exist' % anal_fn)

        for k,line in enumerate(log):
            tt = line.split()

            try:
                int(tt[0])
                begin_line = k
                break
            except:
                continue

        g = log[begin_line:]

        g_data = []
        for line in g:
            a = line.strip('\n').split()
            data = [float(n) for n in a]
            g_data.append(data)
        g_data = np.array(g_data)

        return g_data[:,2:4]

    def get_coordinates(self,trajfile):        
        if not os.path.isfile(trajfile):
            trajfn = os.path.join(self.path,trajfile)

            if not os.path.isfile(trajfn):
                raise Exception(f'The file {trajfile} does not exist')
        
        else:
            trajfn = trajfile
        
        if trajfile[-3:] == 'dcd':
            basenm = self.xyzfile[:-4]
            path = self.path
            cmd = f"cp {path}/{basenm}.xyz {path}/{basenm}.arc"

            os.system(cmd)
            traj = md.load_dcd(trajfn,top=f"{path}/{basenm}.arc")
        elif trajfile[-3:] == 'arc':
            traj = md.load_arc(trajfn)

        self.coords = 10*traj.xyz.copy()

        del traj

    def compute_avg_angle(self,atoms_index=[1,0,2],trajfile=None):

        if self.coords.shape[0] == 0:
            self.get_coordinates(trajfile)

        L = self.coords[self.equil:].shape[0]
        boot = np.random.randint(L,size=int(L/3))

        all_angles = []
        for k,xyz in enumerate(self.coords[boot]):
            a0 = atoms_index[0]
            a1 = atoms_index[1]
            a2 = atoms_index[2]
            fr_angles = [compute_angle(xyz[mol+a0],xyz[mol+a1],xyz[mol+a2]) for mol in range(0,self.Natoms-1,self.Nmonomer)]
            all_angles.append(np.array(fr_angles))

        boot = np.random.randint(L,size=int(L/3))
        for k,xyz in enumerate(self.coords[boot]):
            a0 = atoms_index[0]
            a1 = atoms_index[1]
            a2 = atoms_index[2]
            fr_angles = [compute_angle(xyz[mol+a0],xyz[mol+a1],xyz[mol+a2]) for mol in range(0,self.Natoms-1,self.Nmonomer)]
            all_angles.append(np.array(fr_angles))

        all_angles = np.array(all_angles).flatten()
        self.avg_angle, self.std_angle, err = mean_stderr(all_angles)
        # self.std_angle = np.std(all_angles)


    def compute_avg_bond(self,atoms_index=[0,1],coords=[]):
        if len(coords) == 0:
            coords = self.coords
            if len(self.coords) == 0:
                raise Exception('Load or provide the coordinates')
        Natoms = self.Natoms
        Nmonomer = self.Nmonomer
        eq = self.equil

        L = coords[eq:].shape[0]
        boot = np.random.randint(L,size=int(L/3))

        print(f"Using 2x iterations with {int(L/3)} frames out of {coords.shape}")

        all_bonds = []
        for k,xyz in enumerate(coords[eq:][boot]):
            a0 = atoms_index[0]
            a1 = atoms_index[1]
            fr_bonds = [compute_bond(xyz[mol+a0],xyz[mol+a1]) for mol in range(0,Natoms-1,Nmonomer)]
            all_bonds.append(np.array(fr_bonds))

        boot = np.random.randint(L,size=int(L/3))
        for k,xyz in enumerate(coords[eq:]):
            a0 = atoms_index[0]
            a1 = atoms_index[1]
            fr_bonds = [compute_bond(xyz[mol+a0],xyz[mol+a1]) for mol in range(0,Natoms-1,Nmonomer)]
            all_bonds.append(np.array(fr_bonds))

        all_bonds = np.array(all_bonds).flatten()
        avg_bond = all_bonds.mean()
        std_bond = all_bonds.std()
        
        return avg_bond,std_bond
        # self.std_angle = np.std(all_angles)
