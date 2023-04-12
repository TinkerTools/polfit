import numpy as np

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

class Tinkermol:
    def __init__(self,filenm):
        self.fn = filenm
        self.n_atoms = 0
        self.atom_map = []
        self.masses = []
        self.volume = []
        self.xyz = []
        self.frames = 0
        self.pbc = False
        self.extra = 1
        self.get_xyz()
        
            
    def read_arc_file(self):
        f = open(self.fn)
        data = f.readlines()
        f.close()

        raw_arc = [l.strip('\n') for l in data]

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
        lines_per_frame = (n_atoms+self.extra)

        raw_np = np.array(raw_arc)
        per_frame = np.reshape(raw_np,(int(raw_np.shape[0]/lines_per_frame), int(lines_per_frame)))
        
        return per_frame
    
    def get_xyz(self):
        
        raw_np = self.read_arc_file()
        n_atoms = float(raw_np[0][0].split()[0])
       
        xyz_cords = []
        box_lattice = []
        
        for k,frm in enumerate(raw_np):
            for line in frm[self.extra:]:
                l2 = line.split()[2:5]
                xyz_cords.append([float(a) for a in l2])
                
                if k == 0:
                    self.atom_map.append(line.split()[1])
                    self.masses.append(mw_elements[line.split()[1]])
            if self.pbc:    
                box_lattice.append([float(a) for a in frm[1].split()[0:3]])
        
        self.atom_map = np.array(self.atom_map)
        self.frames = raw_np.shape[0]
        self.xyz = np.reshape(np.array(xyz_cords),(raw_np.shape[0],int(n_atoms),3))
        
        if self.pbc:
            self.volume = np.prod(np.array(box_lattice),axis=1)


### write psi4 input file

def writeinput(xyzfn,options={},sampleinput=None):

    mol = Tinkermol(xyzfn)

    ## options dictionary
    baseoptions = {
        'memory': 1,
        'threads': 1,
        'dimer': 'False',
        'units': 'angstron',
        'basis': '6-311G(d,p)',
        'scf_type': 'df',
        'optimize': None,
        'gradient': None,
        'return_wfn': False,
        'energy': 'mp2',
        'bsse_type': 'cp',
    }

    if len(options.keys() == 0):
        options = baseoptions
    else:
        for key in baseoptions.keys():
            try:
                options[key]
            except:
                options[key] = baseoptions[key]


    #####
    template = f"""memory {options['memory']}
set_num_threads({options['threads']})

molecule mol {{
  0 1
"""

    atomlist = mol.atom_map
    coords = mol.xyz[0]

    if options['dimer'] == 'homo':
        split = int(mol.n_atoms/2)
    elif options['dimer'] == 'mol-water':
        split = int(mol.n_atoms - 3)

    for k, atm in enumerate(atomlist):
        line = f" {atm}{coords[k][0]:18.10f}{coords[k][1]:18.10f}{coords[k][2]:18.10f}\n"

        if k == split:
            line += '--\n'

        template += line
    
    template += f"""
no_com
no_reorient
units {options['units']}
}}\n"""

    setopt = f"""
set {{
    basis       {options['basis']}
    scf_type    {options['scf_type']}
    guess       sad
}}
"""
    template += setopt

    return template

