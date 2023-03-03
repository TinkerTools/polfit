import numpy as np
from numpy import array, zeros, diag, diagflat, dot
from numpy import linalg as LA
import os
import sys
import math
import time
import mdtraj as md
import pickle
from joblib import Parallel, delayed
import multiprocessing, subprocess

num_cores = multiprocessing.cpu_count()

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

class ARC:
    def __init__(self,filenm):
        self.fn = filenm
        self.n_atoms = 0
        self.atom_map = []
        self.connect = []
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
                l3 = line.split()[5:]
                xyz_cords.append([float(a) for a in l2])
                
                if k == 0:
                    self.atom_map.append(line.split()[1])
                    self.connect.append([int(a) for a in l3])
                    self.masses.append(mw_elements[line.split()[1]])
            if self.pbc:    
                box_lattice.append(float(frm[1].split()[0]))
        
        self.atom_map = np.array(self.atom_map)
        self.masses = np.array(self.masses)
        self.connect = np.array(self.connect)
        self.frames = raw_np.shape[0]
        self.xyz = np.reshape(np.array(xyz_cords),(raw_np.shape[0],int(n_atoms),3))
        
        if self.pbc:
            self.volume = np.power(np.array(box_lattice),3)

def save_arc(coords,atmlist,connect,nmol=3):
    natoms = len(atmlist)

    if coords.shape[0] != 6:
        xyz_cords = np.reshape(xyz_cords,(natoms,3))

    base_dimer = f"""{natoms} Dimer"""

    for k,xyzc in enumerate(coords):
        el = atmlist[k]
        con = connect[k]

        cline = ""
        for r in con:
            cline += f'{int(r):5d} '


        line = f"\n{el:2d}    {xyzc[k][0]:18.8f} {xyzc[k][1]:18.8f} {xyzc[k][2]:18.8f} {cline}"
        base_dimer += line

    return base_dimer
    
        
def monomer_xyz(atmlist,coords):
    natoms = len(atmlist)
    base_dimer = f"""{natoms}
monomer"""

    for k,xyzc in enumerate(coords):
        el = atmlist[k]
        line = f"\n{el:2d}    {xyzc[k][0]:18.8f} {xyzc[k][1]:18.8f} {xyzc[k][2]:18.8f}"
        base_dimer+=line

    return base_dimer

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


def compute_integrand(rdist,t,coords,pmi,xcm,pair_index,run_dir,elf_n,do_bkp):
    nm = f"{rdist:.2f}_calc"
    work_dir = f"{run_dir}/{nm}"

    os.chdir(work_dir)
    edim = 0.0
    
    print("Started calculation for dimer for r = %.2f" % (rdist))
    sys.stdout.flush()

    fatm = np.zeros((2,3,3))
    fcmi = np.zeros(3)
    fcmj = np.zeros(3)

    eu    = 0.0
    hist  = 0.0
    fterm = 0.0
    tterm = 0.0

    gasconst=1.987204259e-3
    kt = gasconst * t

    finalize = -1

    if do_bkp:
        os.system("cp dimer.arc dimer.arc.bkp ")
    else:
        ptt = os.getcwd()
        if os.path.isfile("%s/dimer.arc.bkp" % ptt):
            os.system("cp dimer.arc.bkp dimer.arc")
        else:
            os.system("cp dimer.arc dimer.arc.bkp ")
            
    os.system("rm -rf energies.txt number_dimers.txt last_dimer.txt last_line.txt frm_num.txt tmp.txt")
    os.system("rm -rf inter_energy.txt tmp_inter.log good_dimers.arc sed*")
    os.system("wc -l dimer.arc > last_dimer.txt")
    last_frame = 0

    dimer_n = pair_index.shape[0]

    track_frms = dimer_n
    total_good_dimers = 0
    total_bad_dimers = 0
    frames_proc = 0
    runs = 0
    time2 = time.time()
    
    f = open('dimer.arc.bkp')
    dt = f.readlines()
    f.close()

    dimers = [lin.strip('\n') for lin in dt]
    dimers = np.array(dimers)

    dimer_id = []
    while finalize < 0:

        if runs == 0:
            os.system("cp /user/roseane/HIPPO/virial2_data/run_analyze0.sh run_sim.sh")

        commd = "./submit_elf.sh %d" % elf_n
        subprocess.call(commd, shell=True)

        f = open("last_line.txt",'r')
        stat = f.readlines()
        f.close()

        try:
            begin = stat[-1].split()[0]
        except:
            begin = "failed"

        if begin == 'Charge':
            finalize = 1

        else:
            f = open("frm_num.txt",'r')
            stat = f.readlines()
            f.close()

            f = open("number_dimers.txt",'r')
            stat1 = f.readlines()
            f.close()

            n_lines = int(stat1[-1].split()[0])

            if n_lines == 0 or n_lines == 7:
                finalize = 1
                break
            try:
                proc_dimers = int(stat[0].strip("\n").split()[-1])
                line_numb = 7*proc_dimers
            except:
                line_numb = 7
                proc_dimers = 1
                    
            finalize = -1
            os.system("cp /user/roseane/HIPPO/virial2_data/run_analyze.sh run_sim.sh")
            
            # os.system(f"sed -i 's;ZZZd;{line_numb}d;g' run_sim.sh")

            np.savetxt('dimer.arc',dimers[line_numb:],fmt="%s",delimiter='\n')
            dimers = dimers[line_numb:]

                
        runs+=1
    
    f = open('number_dimers.txt')
    dt = f.readlines()
    f.close()

    if len(dt) > 1:
        lines = [int(a.strip('\n').split()[0]) for a in dt]
        diff = []
        for k,d in enumerate(lines[:-1]):
            diff.append(d-lines[k+1])

        diff2 = np.array(diff)
        ids = [diff2[0]]
        for k in range(1,diff2.shape[0]):
            ct = diff2[:k+1].sum()
            ids.append(ct)

        ids = np.array(ids)
        index = ((ids/7) - 1).astype(int)

        f = open('dimer.arc.bkp')
        dt = f.readlines()
        f.close()

        dimers = [lin.strip('\n') for lin in dt]

        total = len(dimers)/7
        dimers = np.reshape(np.array(dimers),(int(total),7))

        good_dimer = np.delete(dimers,index,axis=0)
        np.savetxt('dimer.arc',good_dimer,fmt="%s",delimiter='\n')

        del good_dimer,dimers,dt,diff,ids,diff2,lines
    
    else:
        index = np.array([])
    
    if len(dt) > 1:
        f = open('dimer.arc')
        dt = f.readlines()
        f.close()

        dimers = [lin.strip('\n') for lin in dt]
        dimers = np.array(dimers)
    else:
        f = open('dimer.arc.bkp')
        dt = f.readlines()
        f.close()

        dimers = [lin.strip('\n') for lin in dt]
        dimers = np.array(dimers)

    runs = 0
    finalize = -1
    while finalize < 0:

        if runs == 0:
            os.system("cp /user/roseane/HIPPO/virial2_data/run_sim0.sh run_sim.sh")

        commd = "./submit_elf.sh %d" % elf_n
        subprocess.call(commd, shell=True)

        f = open("last_line.txt",'r')
        stat = f.readlines()
        f.close()


        try:
            begin = stat[-1].split()[0]
            begin10 = stat[0].split()[0]
        except:
            begin = "failed"
            begin10 = "None"


        if begin == 'Anlyt':
            finalize = 1

        else:
            if begin10 == "Additional" and begin == "Tinker":
                line_numb = 7
                proc_dimers = 1

            else:
                f = open("frm_num.txt",'r')
                stat = f.readlines()
                f.close()

                f = open("number_dimers.txt",'r')
                stat1 = f.readlines()
                f.close()

                n_lines = int(stat1[-1].split()[0])

                if n_lines == 0 or n_lines == 7:
                    finalize = 1
                    break
                try:
                    proc_dimers = int(stat[0].strip("\n").split()[-1])
                    line_numb = 7*proc_dimers
                except:
                    line_numb = 7
                    proc_dimers = 1
                    
            finalize = -1
            os.system("cp /user/roseane/HIPPO/virial2_data/run_sim.sh run_sim.sh")
            # os.system("sed -i 's;ZZZd;%dd;g' run_sim.sh" % line_numb)

            np.savetxt('dimer.arc',dimers[line_numb:],fmt="%s",delimiter='\n')
            dimers = dimers[line_numb:]

        runs+=1
    
    if index.shape[0] > 0:
        pair_ind_new = np.delete(pair_index,index,axis=0)
    else:
        pair_ind_new = np.copy(pair_index)
    
    failed_pairs = []
    f = open("number_dimers.txt", 'r')
    data = f.readlines()
    f.close()
    
    if len(data) > 1:
        total_dimers = int(data[0].split()[0])
        for line in data[2:]:
            num = float(line.split()[0])

            diff = (total_dimers-num)/7

            failed_pairs.append(int(diff-1))

    #failed_pairs = np.array(failed_pairs)
           
    f = open("inter_energy.txt", 'r')
    data = f.readlines()
    f.close()
    
    inter_energy = [float(a.split()[-2]) for a in data]
    inter_energy = np.array(inter_energy)
    
    if len(failed_pairs) > 0:
        f2 = np.array(failed_pairs)
        inter_energy = np.delete(inter_energy,f2,axis=0)
        
        all_inds = list(range(pair_ind_new.shape[0]))
        for a in failed_pairs:
            all_inds.remove(a)
    
    else:
        all_inds = list(range(pair_ind_new.shape[0]))
        
    forces = []
    energies = []

    f = open("energies.txt", 'r')
    data = f.readlines()
    f.close()

    begin_line = np.array([a[1:16] for a in data])
    inds = np.where(begin_line == "Total Potential")[0]

    big_forces = []
    energies = []
    forces = []
    hist = 0
    fterm = 0
    tterm = 0
    eu = 0
    
    for q,k in enumerate(inds):
        line = data[k]
        e = False
        try:
            e = float(line.split()[-2])
        except:
            e = False

        if e:
            lin2 = [a.split()[2:5] for a in data[k+136:k+136+6]]
            ff = []
            for lin in lin2:
                try:
                    test = float(lin[0])
                    ff.append([float(b) for b in lin])
                except:
                    break

            if len(ff) == 6 and e > -10.0:
                forces.append(ff)
                energies.append(e)
            else:
                big_forces.append(all_inds[q])

    energies = np.array(energies)
    forces = np.array(forces)
    
    del data,inds,begin_line
    
    if len(big_forces) > 0:
        b2 = np.array(big_forces)
        inter_energy = np.delete(inter_energy,b2,axis=0)
    
        big_forces = np.array(big_forces)
    
        pair_ind_new = np.delete(pair_ind_new,big_forces.astype(int),axis=0)
    
    dict_res = {'energy': [],
                'pmi': [],
                'xyz': [],
                'xcm': [],
                'forces': [],
                'torques': []}
    
    for q in range(forces.shape[0]):
        i = pair_ind_new[q][0]
        j = pair_ind_new[q][1]

        x = np.copy(coords[i])
        xicm = np.copy(xcm[i])

        fatm = np.reshape(forces[q],(2,3,3))
        epot = np.copy(energies[q])
        edim = np.copy(inter_energy[q])
        
        
        if edim > -10.0:           
            bolt = np.exp(-edim/kt)
                        
            trq = np.zeros((3))

            fcmi = np.array(fatm[0]).sum(axis=0)
            fcmj = np.array(fatm[1]).sum(axis=0)

            rcmx = np.zeros((3,3))
            rcmx[:,0] = x[:,0] - xicm[0]
            rcmx[:,1] = x[:,1] - xicm[1]
            rcmx[:,2] = x[:,2] - xicm[2]

            trqx = rcmx[:,1]*fatm[0][:,2]-rcmx[:,2]*fatm[0][:,1]
            trqy = rcmx[:,2]*fatm[0][:,0]-rcmx[:,0]*fatm[0][:,2]
            trqz = rcmx[:,0]*fatm[0][:,1]-rcmx[:,1]*fatm[0][:,0]

            trq[0]+=trqx.sum()
            trq[1]+=trqy.sum()
            trq[2]+=trqz.sum()
            
            dict_res['energy'].append(edim)
            dict_res['xyz'].append(x)
            dict_res['xcm'].append(xicm)
            dict_res['pmi'].append(pmi[i])
            dict_res['forces'].append(fcmi)
            dict_res['torques'].append(trq)

            if rdist < 2.5 and bolt > 1e2:
                continue
            elif rdist > 2.5 and bolt > 1e4:
                continue
                
            hist += 1
            eu += (bolt - 1) 
            ff = np.power(fcmi,2)
            fterm += bolt*(ff.sum())
            tt = np.power(trq[0],2)/pmi[i][0]+np.power(trq[1],2)/pmi[i][1]+np.power(trq[2],2)/pmi[i][2]
            tterm += bolt*tt
            
    if hist > 0:
        eterm1 = (eu/hist)
        fterm1 = (fterm/hist)
        tterm1 = (tterm/hist)

        os.system("mkdir -p %s/results" % run_dir)
        np.save('%s/results/%s.npy' % (run_dir,nm),np.array([rdist,eterm1,fterm1,tterm1]))
        save_pickle(dict_res,'%s/results/all_data_%s.npy' % (run_dir,nm))
        
        time3 = time.time()
        print("Finished r = %.2f in %.1f seconds (%.1f hours)" % (rdist,time3-time2,(time3-time2)/3600.0))
        return np.array([rdist,eterm1,fterm1,tterm1])
    else:
        time3 = time.time()
        print("Finished r = %.2f in %.1f seconds (%.1f hours)" % (rdist,time3-time2,(time3-time2)/3600.0))
        return np.array([rdist,0.0,0.0,0.0])

def main():
    try:
        xyzfile = str(sys.argv[1])

        if xyzfile[-4:] != ".xyz":
            xyzfile += ".xyz"
    except:
        xyzfile = "liquid.xyz"

    try:
        logfile = str(sys.argv[2])

    except:
        logfile = "liquid.log"

    try:
        analysis = str(sys.argv[3])

    except:
        analysis = "analysis.log"

    if os.path.isfile(xyzfile):
        process = subprocess.Popen("head -2 %s" % xyzfile, 
            stdout=subprocess.PIPE,shell=True,encoding='utf8')

        output = process.stdout.readlines()
        stdout = process.communicate()[0]

        N = int(output[0].split()[0])

        lz = float(output[1].split()[2])
        vol = lz*float(output[1].split()[0])*float(output[1].split()[1])
    else:
        print("The given xyz file does not exist\n")
        sys.stdout.flush()
        return

    virial_tensor = []
    t0 = time.time()
    if os.path.isfile(analysis):

        print("Start reading virial file...\n")
        sys.stdout.flush()

        f = open(analysis,'r')
        pe_data = f.readlines()
        f.close()
        
        sys.stdout.flush()
   
        begin_lines = [dt[0:4] for dt in pe_data]
        begin_lines = np.array(begin_lines)
        pe_data = np.array(pe_data)
        pe_ind = np.where(begin_lines==' Int')[0]
        
        sys.stdout.flush()
        for ind in pe_ind:
            tt =[]
            tv0 = pe_data[ind].strip('\n').split()[-3:]
            tt.append([float(a) for a in tv0])
            tv0 = pe_data[ind+1].strip('\n').split()[-3:]
            tt.append([float(a) for a in tv0])
            tv0 = pe_data[ind+2].strip('\n').split()[-3:]
            tt.append([float(a) for a in tv0])
            virial_tensor.append(tt)
        
        sys.stdout.flush()
        del begin_lines, pe_data, pe_ind

    else:
        return
            
    virial_tensor = np.array(virial_tensor)
    np.save("virial.npy",virial_tensor)

    print("Finished processing virial file...\n")
    sys.stdout.flush()
    
    velocity_file = xyzfile[:-4]+'.vel'
    if os.path.isfile(velocity_file):
        masses = {'H':1.0078250321,'O':15.9949146221}
        
        print("Start reading velocity file...\n")
        sys.stdout.flush()

        process = subprocess.Popen("tail -%d %s" % (N+1,velocity_file), 
            stdout=subprocess.PIPE,shell=True,encoding='utf8')

        output = process.stdout.readlines()
        stdout = process.communicate()[0]

        atm_type = [line.split()[1] for line in output[1:]]

        m_atms = []
        for at in range(len(atm_type)):
            atm = atm_type[at]
            m = masses[atm]
            m_atms.append(m)

        m_atms = np.array(m_atms)

    else:
        return

    print("Finished processing velocity file...\n")
    sys.stdout.flush()

    pressure_tensor = np.zeros((virial_tensor.shape[0],3,3))
    #velocity_data = np.zeros((virial_tensor.shape[0],N,3))

    print("Calculating pressure tensor...\n")
    sys.stdout.flush()

    NA=6.02214129*(1e23)

    num_cores = multiprocessing.cpu_count()

    
    #f = open(velocity_file,'r')
    #vel_data = f.readlines()
    #f.close()

    t1 = time.time()

    print("It took %d seconds to open velocity file..." % (t1-t0))
    sys.stdout.flush()
    split_frms = list(range(0,virial_tensor.shape[0],int(virial_tensor.shape[0]/num_cores)))
    split_frms.append(virial_tensor.shape[0])
    
    def calc_pres_tensor(k):
        #data = np.copy(vel_data[k:k+N+1])
        f1 = split_frms[k]
        f2 = split_frms[k+1] 
        first = 1+f1*(N+1)
        last = 1+(f2)*(N+1)-1
        cmd = "sed -n '%d,%dp;%dq' %s" % (first,last,last+1,velocity_file)
        process = subprocess.Popen(cmd, 
            stdout=subprocess.PIPE,shell=True,encoding='utf8')

        output = process.stdout.readlines()
        stdout = process.communicate()[0]

        for i,fr in enumerate(range(first-1,last-1,N+1)):
            velocity = []
            
            #print(i,fr+1,fr+N+1,len(output))
            #sys.stdout.flush()
            fr_ind = list(range(0,len(output),N+1))

            for line in output[fr_ind[i]+1:fr_ind[i]+N+1]:
                line2 = line.strip('\n').split()
                v = [float(a.replace('D','e')) for a in line2[-3:]]
                velocity.append(v)

            V = np.array(velocity)
            #print(V.shape)
            sys.stdout.flush()
            del velocity
            pres = np.zeros((3,3))
                        
            fr_n = f1+i
            VR = 4184.0*virial_tensor[fr_n]
            
            for kk in range(3):
                mvv = m_atms*V[:,kk]*V[:,kk]
                mvv1 = mvv.sum()
                pres[kk,kk]+= (10*mvv1 - VR[kk,kk])
                
                
            for kk in range(1,3):
                mvv = m_atms*V[:,0]*V[:,kk]
                mvv1 = mvv.sum()
                pres[0,kk]+= (10*mvv1 - VR[0,kk])
                pres[kk,0]+= (10*mvv1 - VR[0,kk])
                
            #[1,2]
            mvv = m_atms*V[:,1]*V[:,2]
            mvv1 = mvv.sum()
            pres[1,2]+= (10*mvv1 - VR[1,2])
            pres[2,1]+= (10*mvv1 - VR[1,2])

            pres *= ((1e30)/(NA*vol))
            pressure_tensor[fr_n] = np.array(pres)
        #pressure_tensor = np.array(pressure_tensor) # (J/m3)

            if fr_n % 100 == 0:
                print("Finished %d..." % fr_n)
                sys.stdout.flush()
            del V, pres, mvv, mvv1
        del output

    Parallel(n_jobs=1)(delayed(calc_pres_tensor)(k) for k in range(len(split_frms[:-1])))
    #del vel_data

    t2 = time.time()
    print("Finished calculation. Saving data...\n")
    print("It took %d seconds to finish calculation..." % (t2-t1))
    sys.stdout.flush()

    #velocity_data = np.array(res[:,1])
    #frms = np.array(res[:,0])

    #print("%d %d %d..." % (fmrs[0],fmrs[-1],fmrs[499]))
    #sys.stdout.flush()
    #np.save("velocity.npy",velocity_data)  
    np.save("pressure.npy",pressure_tensor)

if __name__ == "__main__":
    main()


