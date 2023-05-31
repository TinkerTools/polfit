from ast import Delete
import numpy as np
import pickle
import os
from . import prmedit

avogadro=6.02214076e23
lightspd=2.99792458e-2
boltzmann=0.831446262
gasconst=1.987204259e-3
elemchg=1.602176634e-19
vacperm=8.854187817e-12
emass=5.4857990907e-4
planck=6.62607015e-34
joule=4.1840
ekcal=4.1840e2
bohr=0.52917721067
hartree=627.5094736
evolt=27.21138602
efreq=2.194746314e5
coulomb=332.063713
debye=4.80321
prescon=6.85684112e4

water_mw = 18.015

mw_elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941,
               'Be': 9.012182, 'B': 10.811, 'C': 12.0107,
               'N': 14.0067,'O': 15.9994,'F': 18.9984032,
               'Ne': 20.1797,'Na': 22.98976928,'Mg': 24.305,
               'Al': 26.9815386,'Si': 28.0855,'P': 30.973762,
               'S': 32.065,'Cl': 35.453,'Ar': 39.948,
               'K': 39.0983,'Ca': 40.078,'Sc': 44.955912,
               'Ti': 47.867,'V': 50.9415, 'Cr': 51.9961,
               'Mn': 54.938045,'Fe': 55.845,'Co': 58.933195,
               'Ni': 58.6934,'Cu': 63.546,'Zn': 65.409,
               'Ga': 69.723,'Ge': 72.64,'As': 74.9216,
               'Se': 78.96,'Br': 79.904,'Kr': 83.798,
               'Rb': 85.4678,'Sr': 87.62, 'Y': 88.90585,
               'Zr': 91.224,'Nb': 92.90638,'Mo': 95.94,
               'Tc': 98.9063,'Ru': 101.07,'Rh': 102.9055,
               'Pd': 106.42,'Ag': 107.8682,'Cd': 112.411,
               'In': 114.818,'Sn': 118.71,'Sb': 121.76,
               'Te': 127.6,'I': 126.90447,'Xe': 131.293,
               'Cs': 132.9054519,'Ba': 137.327,'La': 138.90547,
               'Ce': 140.116,'Pr': 140.90465,'Nd': 144.242,
               'Pm': 146.9151,'Sm': 150.36,'Eu': 151.964,
               'Gd': 157.25,'Tb': 158.92535,'Dy': 162.5,
               'Ho': 164.93032,'Er': 167.259,'Tm': 168.93421,
               'Yb': 173.04,'Lu': 174.967,'Hf': 178.49,
               'Ta': 180.9479,'W': 183.84,'Re': 186.207,
               'Os': 190.23,'Ir': 192.217,'Pt': 195.084,
               'Au': 196.966569,'Hg': 200.59,'Tl': 204.3833,
               'Pb': 207.2,'Bi': 208.9804,'Po': 208.9824,
               'At': 209.9871,'Rn': 222.0176,'Fr': 223.0197,
               'Ra': 226.0254,'Ac': 227.0278,'Th': 232.03806,
               'Pa': 231.03588,'U': 238.02891,'Np': 237.0482,
               'Pu': 244.0642,'Am': 243.0614,'Cm': 247.0703,
               'Bk': 247.0703,'Cf': 251.0796,'Es': 252.0829,
               'Fm': 257.0951,'Md': 258.0951,'No': 259.1009,
               'Lr': 262,'Rf': 267,'Db': 268,'Sg': 271,
               'Bh': 270,'Hs': 269,'Mt': 278,'Ds': 281,
               'Rg': 281,'Cn': 285,'Nh': 284,'Fl': 289,
               'Mc': 289,'Lv': 292,'Ts': 294,'Og': 294,
               'ZERO': 0} 

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

def get_mass(xyz_fn): 
    """return the total mass within a .xyz file """
    f = open(xyz_fn)
    lines = f.readlines()
    f.close()

    try:
        elements = [a.split()[1] for a in lines[1:]]
        float(mw_elements[elements[0]])

    except:
        elements = [a.split()[1] for a in lines[2:]]
        float(mw_elements[elements[0]])
    total_mass = 0
    for X in elements:
        total_mass += float(mw_elements[X])  

    return total_mass

def count_atoms(fn):
    """return the number of atoms in .xyz file """
    f = open(fn)
    dt = f.readline()
    f.close()

    c = int(dt.split()[0])

    return c   

def compute_center_of_mass(fname="dimer.arc",natms=[]):
    """Give a dimer xyz or arc, returns center of mass of each monomer and the structure center of mass"""
    
    arcf = ARC(fname)
    arcf.read_arc_file()
    coords = arcf.arcxyz
    
    if len(natms) == 0:
        natm = coords[0].shape[0]
        n1 = int(natm/2)
        n2 = n1
    else:
        n1 = natms[0]
        n2 = natms[1]
    
    xyz0,atommap = prmedit.read_xyz_file(fname)
    masses = np.array([mw_elements[a[1]] for a in atommap])
    
    coords = arcf.arcxyz
    cm = np.zeros((coords.shape))
    for i in range(3):
        cm[:,:,i] += coords[:,:,i] * masses
    
    cm1 = cm[:,:n1,:].sum(axis=1)/masses[:n1].sum()
    cm2 = cm[:,n1:,:].sum(axis=1)/masses[n1:].sum()
    fcm = cm.sum(axis=1)/masses.sum()
    
    return cm1,cm2,fcm

def num_molecules(box_size,mw,dens):
    """Needs (box_size (Angstroms),molecular_weight ((g/mol),density (g/ml))
    returns number of molecules"""
    A=6.02214129*(1e23)
    density=dens/(1e24)
    mol_w = mw/A
    num_molecules = (np.power(box_size,3)*density)/mol_w
    return np.rint(num_molecules)

def calc_density(box_size,n_mol,mw):
    """Needs (box_size (Angstroms),number_of_molecules,molecular_weight (g/mol))
    returns density in g/mol"""
    A=6.02214129*(1e23)
    mol_w = mw/A
    dens = (n_mol*mol_w)/(np.power(box_size,3))
    return dens*(1e24)
    
def calc_box_s(n_mol,mw,dens):
    """Needs (number_of_molecular_weight (g/mol),density (g/ml))
    returns box_size in Angstroms"""
    A=6.02214129*(1e23)
    density=dens/(1e24)
    mol_w = mw/A
    box_s = np.cbrt((n_mol*mol_w)/density)
    return box_s

def num_steps(dt,t):
    """ Needs (time_step (fs), desired_length (ns))
    return number of steps"""
    tfs = t*1e6
    dT = dt*1e0
    return tfs/dT

def time_ps(dt,step):
    """ Needs (time_step (fs), number_of_steps)
    return total time in ps"""
    tfs = step*dt*1e0
    return tfs/1e3

def deviation(dens,true_dens):
    dev = abs(dens-true_dens)/true_dens
    return dev

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def read_all_arc(xyzfn):
    if isinstance(xyzfn,list):
        thefile = xyzfn
        openfile = False
    else:
        test = xyzfn.split('\n')
        if len(test) > 2:
            thefile = xyzfn.split('\n')
            openfile = False
        else:
            if os.path.isfile(xyzfn):
                openfile = True
            else:
                print("XYZ file does not exist!")
                return

    if openfile:
        thefile = []
        lread = 0
        f = open(xyzfn)
        for a in range(3):
            thefile += f.readline()
            lread+=1
        f.close()

        thefile = "".join(thefile)
        thefile = thefile.split('\n')
    
    test = thefile[2].split()[0]
    if test.isdigit():
        st = 1
        ni = 1
    else:
        st = 2
        ni = 0

    natms = int(thefile[0].split()[0])
    nlines = st+natms

    if openfile:
        lread = 0
        full = []
        with open(xyzfn) as thefile:
            for line in thefile:
                if line != '\n':
                    full.append(line)
                    lread += 1
        
        nframes = lread/nlines
        thefile = full
    else:
        thefile = [a for a in thefile if a != '\n']
        nframes = len(thefile)/nlines
    
    return np.reshape(np.array(thefile),(int(nframes),nlines))

type_map = {"CA": "C", "HA": "H"}
class ARC:
    def __init__(self,filenm):
        self.fn = filenm
        self.n_atoms = 0
        self.atom_map = []
        self.volume = []
        self.lattice = []
        self.xyz = []
        self.nframes = 0
        # self.read_xyz()
        self.pbc = False
        self.extra = 1
            
    def read_arc_file(self):
        ## check xyz file first
        isARC = False
        if self.fn[-3:] == 'arc':
            finit = self.fn[:-3]+'xyz'
            isARC = True
        elif self.fn[-3:] == 'xyz':
            finit = self.fn

        self.arcxyz = []
        self.lattice = []
        if not os.path.isfile(finit) and isARC:
            full_arc = read_all_arc(self.fn)
            for frm in full_arc:
                xyz = self.process_data(frm)
                self.arcxyz.append(xyz)
                self.raw = frm

            self.arcxyz = np.array(self.arcxyz)
            self.nframes = self.arcxyz.shape[0]
            self.lattice = np.array(self.lattice)

        elif os.path.isfile(finit) and isARC:
            self.read_xyz(finit)
            chunk = os.path.getsize(finit)
            with open(self.fn) as f:
                aa = read_in_chunks(f,chunk)
                for frm in aa:
                    self.raw = frm
                    xyz = self.process_data(frm)
                    self.arcxyz.append(xyz)

                self.arcxyz = np.array(self.arcxyz)
                self.nframes = self.arcxyz.shape[0]
            self.lattice = np.array(self.lattice)
        elif os.path.isfile(finit) and not isARC:
            self.read_xyz(finit)
        else:
            return

    def process_data(self,frm):
        try:
            data = frm.split('\n')
        except:
            data = frm
        
        test = data[2].split()[0]
        if test.isdigit():
            ni = 1

            test2 = data[1].split()[1]
            try:
                float(test2)
                self.extra = 2
                st = 2
            except:
                st = 1
                self.extra = 1
            
        else:
            self.extra = 2
            st = 2
            ni = 0

        if ni == 1 and st == 2:
            s = data[1].split()
            self.lattice.append([float(a) for a in s])
        
        self.title = data[0]
        xyz = []
        for line in data[st:]:
            s = line.split()
            if len(s) == 0:
                continue
            xyz.append([float(a) for a in s[ni+1:ni+4]])

        return xyz

    def read_xyz(self,xyzfn=None):   
        if xyzfn != None: 
            f = open(xyzfn)
        else:
            f = open(self.fn)
        data = f.readlines()
        f.close()
        n_atoms = float(data[0].split()[0])
        
        #
        try:
            lt = float(data[1].split()[1])
            self.extra = 2
            self.pbc = True
        except:
            self.pbc = False
            self.extra = 1
        
        self.n_atoms = n_atoms
        self.lines_per_frame = (n_atoms+self.extra)


        if self.pbc:
            s = data[1].split()
            box_lattice = [float(a) for a in s]
        
        xyz_cords = []
        for line in data[self.extra:]:
            s = line.split()
            if len(s) == 0:
                continue
            xyz_cords.append([float(a) for a in s[2:5]])
            
            try:
                m = float(mw_elements[s[1]])
            except:
                e1 = type_map[s[1]]
                m = float(mw_elements[e1])

            self.atom_map.append(m)
        
        self.atom_map = np.array(self.atom_map)
        self.xyz = np.array(xyz_cords)
        self.lattice.append(box_lattice)
        self.lattice = np.array(self.lattice)
            
    def compute_volume(self):        
        self.volume = []
        if self.lattice.shape[0] == 0:
            return

        for dt in self.lattice:
            if int(dt[3]) == 90 and int(dt[4]) == 90 and int(dt[5]) == 90:
                self.volume.append(dt[0]*dt[1]*dt[2])

    def save_arc(self,frms,fnout=""):
        final = ""
        for frm in self.arcxyz[frms]:
            final += self.raw[0]
            if self.extra == 2:
                final += self.raw[1]
            
            test = self.raw[2].split()
            ni = 0
            if test[0].isdigit():
                ni = 1
            
            atommap = [a.split()[ni] for a in self.raw[self.extra:] if a != '\n']
            template = [a.split()[5:] for a in self.raw[self.extra:] if a != '\n']
            for k in range(frm.shape[0]):
                el = atommap[k]
                s = frm[k]
                typ = int(template[k][0])
                if ni == 1:
                    final += f"{k+1:6d}  {el:<2s} {s[0]:12.6f} {s[1]:12.6f} {s[2]:12.6f} {typ:5d}"

                    if len(template) > 1:
                        con = [int(a) for a in template[k][1:]]   
                        for a in con:
                            final += f" {a:5d}"
                    
                    final += '\n'
                else:
                    final += f"{el:<2s} {s[0]:12.6f} {s[1]:12.6f} {s[2]:12.6f}\n"
            
        if len(fnout) > 0:
            with open(fnout,'w') as outfile:
                outfile.write(final)  
            return
        else:
            return final