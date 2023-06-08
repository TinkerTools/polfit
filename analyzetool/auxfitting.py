from cmath import isnan
from curses import KEY_FIND
from genericpath import isfile
from mimetypes import init
from re import I
from turtle import get_poly
from xml.etree.ElementPath import get_parent_map
import numpy as np
import os
import sys
import math
import scipy.optimize as optimize
from time import gmtime, strftime
import warnings
warnings.filterwarnings('ignore') # make the notebook nicer
import shutil 
from collections import namedtuple
import subprocess, threading
import pickle
from pickle import Pickler, Unpickler
from collections import OrderedDict, defaultdict
import analyzetool
import analyzetool.gas as gasAnalyze
import analyzetool.liquid as liqAnalyze
import pkg_resources
import datetime, time
import signal

R=1.9872036E-3 #Kcal/K.mol
KB = 1.38064852E-16 #J/K
KB_J = 1.38064852E-23 #J/K
E0 = 8.854187817620E-12
A=6.02214129*(1E23)
charge = 1.602176634E-19
P_PA = 101325.0

econv = 1.64877727436e-41 ## From atomic electric polarizability to SI units
conv = (1e30)/(4*np.pi*E0) ## From SI units to A^3 
                           ## (chemeurope.com/en/encyclopedia/Polarizability.html)
elec_pol_A3 = conv*econv


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

def round1(a,dec=6):
    return np.round(a,dec)

def count_atoms(fn):
    f = open(fn)
    dt = f.readlines()
    f.close()

    test = dt[2].split()

    if isinstance(test[1], str):
        n = 1
    else:
        n = 2

    c = 0
    for l in dt[n:]:
        if len(l.split()) >= 6:
            c+=1

    return c    

def copy_files(src,destdir):
    fname = os.path.basename(src)

    if os.path.isdir(destdir):
        destnm = f"{destdir}/{fname}"
    else:
        destnm = destdir
    if os.path.isfile(src):
        if os.path.isfile(destnm):
            os.remove(destnm)
    else:
        return
    
    shutil.copy(src,destnm)
    return

def get_last_frame(fname):    
    if 'gas2.log' in fname:
        cmd = f"""grep "Analysis for" {fname} | wc -l"""
    else:
        cmd = f"""grep "Current Time" {fname} | wc -l"""
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True,encoding='utf8')
    output = process.stdout.readline()
    stdout = process.communicate()[0]

    try:
        n_lines = int(output.split()[0])
    except:
        n_lines = 0
    
    return n_lines

energy_terms = np.array(['Stretching', 'Bending', 'Stretch-Bend', 'Bend', 'Angle',
       'Torsion', 'Waals',
       'Repulsion', 'Dispersion', 'Multipoles', 'Polarization',
       'Transfer'], dtype='<U12')


#######################
HOME = os.path.expanduser('~')
smallmoldir = "/work/roseane/HIPPO/small_molecules"
datadir = f"{smallmoldir}/org_molecules/reference-data"
tinkerpath = f'{HOME}/tinker'
#######################

## Process start parameter files, save dictionary with starting parameters
## Copy parameter file to fit directory and water parameters
## Create parameters to fit array
## Convert parameters to fit back to dictionary
## Make keyfile based on dictionary
## Get experimental data, put it in array and save locally
##

class Auxfit(object):

    def __init__(self,base_dir,molnumber):
        self.basedir = base_dir
        self.molnumber = molnumber
        self.tinkerpath = f'{tinkerpath}/bin'
        self.initpotrms = 0

        self.nsteps = 250000
        self.nsteps_gas = 2000000
        self.equil = 200
        self.molpol = 0
        self.progfile = f'{base_dir}/{molnumber}/progress.pickle'
        self.rungas = True

        global i 
        i = 0
        self.optimizer = 'genetic'
        self.log = []

        self.termfit = ["bond-force","angle-force",'chgpen','dispersion','repulsion',
                'polarize','bndcflux','angcflux',
                'chgtrn','multipole']

        self.useliqdyn = False
        self.do_dimers = False
        self.do_clusters = False
        self.do_sapt_dimers = False
        self.do_ccsdt_dimers = False
        self.dumpdata = True
        self.fitliq = False
        self.usedatafile = False
        self.computeall = False

        self.datadir = datadir
        self.smallmoldir = smallmoldir
    def prepare_directories(self):
        n = self.molnumber
        potdir = f"{self.basedir}/{n}/potential-test"
        poldir = f"{self.basedir}/{n}/mol-polarize"
        dimerdir = f"{self.basedir}/{n}/dimer"
        liqdir = f"{self.basedir}/{n}/liquid"
        refliqdir = f"{self.basedir}/{n}/ref_liquid"
        dumpdir = f"{self.basedir}/{n}/dumpdata"

        folders = [potdir,poldir,dimerdir,liqdir,dumpdir]
                
        prmfn = f"{self.datadir}/prmfiles/{n}.prm"
        waterprm = f"{self.datadir}/prmfiles/water-prms.prm"
        xyzpath = f"{self.datadir}/boxes/{n}"
        qmpath = f"{self.datadir}/qm-calc/{n}"
        elecpot = f"{self.datadir}/elec-pot/{n}"
        molpol = f"{self.datadir}/mol-polarize/{n}"

        for k,dest in enumerate(folders):
            if not os.path.isdir(dest):
                os.makedirs(dest)

            if dest != dumpdir:
                copy_files(prmfn,dest)
            
            if dest == liqdir:
                copy_files(f"{xyzpath}/monomer.xyz",f"{dest}/gas.xyz")
                copy_files(f"{xyzpath}/monomer.key",f"{dest}/gas.key")
                copy_files(f"{xyzpath}/liquid.xyz",dest)
                copy_files(f"{xyzpath}/liquid.key",dest)
            if dest == poldir:
                copy_files(f"{molpol}/monomer.xyz",dest)
            if dest == dimerdir:
                os.system(f"cat {waterprm} >> {dest}/{n}.prm")
            if dest == potdir:
                copy_files(f"{elecpot}/monomer.pot",dest)
                copy_files(f"{elecpot}/monomer.xyz",dest)
                
        self.Natoms = count_atoms(f"{xyzpath}/monomer.xyz")
        self.Natomsbox = count_atoms(f"{xyzpath}/liquid.xyz")
        self.Nmol = int(self.Natomsbox/self.Natoms)

        ## Load reference data
        self.refmolpol = np.load(f"{molpol}/{n}-poleigen.npy")
        self.refmolpol *= elec_pol_A3

        if os.path.isfile(f"{self.datadir}/database-info/molinfo_dict.pickle"):
            self.molinfo = load_pickle(f"{self.datadir}/database-info/molinfo_dict.pickle")
            self.liquid_fitproperties()
        else:
            self.molinfo = {}

        # Directory to dump data from the run
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M')
        self.dumpfile = f"{dumpdir}/log-{st}.pickle"
        self.dumpresult = f"{dumpdir}/result-{st}.pickle"

        self.gasdcd = ""
        gas = f"{refliqdir}/gas.dcd"
        if os.path.isfile(gas):
            self.gasdcd = gas

    def process_prm(self,prmfn=None):
        n = self.molnumber
        if prmfn == None:
            prmfn = f"{self.datadir}/prmfiles/{n}.prm"
        
        f = open(prmfn)
        prmfile = f.readlines()
        f.close()
        
        prmfile = np.array(prmfile)
        nstart = 0
        for k,line in enumerate(prmfile):
            if line[:6] == 'atom  ':
                nstart = k
                break

        prmfile = prmfile[prmfile != '\n']
        
        bg = np.array([a[:5] for a in prmfile])
        atms = prmfile[bg=='atom ']
        bnd = prmfile[nstart:][bg[nstart:]=='bond ']
        angl = prmfile[nstart:][bg[nstart:]=='angle']
        strb = prmfile[nstart:][bg[nstart:]=='strbn']
        opbe = prmfile[nstart:][bg[nstart:]=='opben']
        tors = prmfile[nstart:][bg[nstart:]=='torsi']
        pol = prmfile[nstart:][bg[nstart:]=='polar']  
        cpen = prmfile[nstart:][bg[nstart:]=='chgpe']
        disp = prmfile[nstart:][bg[nstart:]=='dispe']
        rep = prmfile[nstart:][bg[nstart:]=='repul']
        ctrn = prmfile[nstart:][bg[nstart:]=='chgtr']
        bflx = prmfile[nstart:][bg[nstart:]=='bndcf']
        aflx = prmfile[nstart:][bg[nstart:]=='angcf']
        exchp = prmfile[nstart:][bg[nstart:]=='exchp']
        
        minds = np.where('multi' == bg)[0]
        
        mlines = prmfile[minds[0]:minds[-1]+5] 
        
        # atom types
        atoms = []
        atmtyp = []
        bond = [[],[],[]]
        angle = [[],[],[],[]]
        strbnd = [[],[],[]]
        opbend = [[],[]]
        torsion = [[],[]]
        multipoles = [[],[]]
        bndcflux = [[],[]]
        angcflux = [[],[]]
        exchpol = []

        typcls = {}
        tclasses = []
        for k,lin in enumerate(atms):
            t = int(lin.split()[1])
            cl = int(lin.split()[2])
            typcls[t] = cl
            atmtyp.append(t)
            tclasses.append(cl)

            nm = lin.split('"')
            atmline = nm[0].split()[2:] + [nm[1]] + nm[2].split()
            atoms.append(atmline)
        
        tclasses = list(set(tclasses))
        tclasses = sorted(tclasses)
        nclas = len(tclasses)
        clspos = {t:i for i,t in enumerate(tclasses)}

        atmtyp = sorted(atmtyp)
        atmpos = {t:i for i,t in enumerate(atmtyp)}
        natms = len(atmtyp)
        #sort
        chgpen = np.zeros((nclas,2))
        dispersion = np.zeros((nclas))
        repulsion = np.zeros((nclas,3))
        chgtrn = np.zeros((nclas,2))
        polarize = [[0]*natms,[0]*natms]
        for k in range(nclas):
            tl = int(cpen[k].split()[1])
            i = clspos[tl]
            v1 = float(cpen[k].split()[2])
            v2 = float(cpen[k].split()[3])
            chgpen[i] = [v1,v2]
            ##
            tl = int(disp[k].split()[1])
            i = clspos[tl]
            v1 = float(disp[k].split()[2])
            dispersion[i] = v1
            ##
            tl = int(rep[k].split()[1])
            i = clspos[tl]
            v1 = float(rep[k].split()[2])
            v2 = float(rep[k].split()[3])
            v3 = float(rep[k].split()[4])
            repulsion[i] = [v1,v2,v3]
            ##
            tl = int(ctrn[k].split()[1])
            i = clspos[tl]
            v1 = float(ctrn[k].split()[2])
            v2 = float(ctrn[k].split()[3])
            chgtrn[i] = [v1,v2]
        
        for k in range(natms):
            s = pol[k].split()
            i = atmpos[int(s[1])]
            v1 = float(s[2])
            polarize[0][i] = v1
            polarize[1][i] = s[3:]        
            ##
        
        multipole_rules = {}
        for i,line in enumerate(mlines[::5]):
            s2 = line.split('#')
            s1 = s2[0]
            typs = s1.split()[1:-1]
            vals = [float(s1.split()[-1])]

            nm = f"c{i+1}"
            if len(s2) > 1:
                nn = s2[1].split()
                nmf = nn[0]
                if nmf == nm:
                    multipole_rules[nm] = float(s1.split()[-1])
                else:
                    multipole_rules[nm] = "".join(nn)
            else:
                multipole_rules[nm] = float(s1.split()[-1])
            
            for z,k in enumerate(mlines[i*5+1:i*5+5]):
                s = k.split()
                if z == 3:
                    vals += [float(a) for a in s[:-1]]
                else:
                    vals += [float(a) for a in s]
            multipoles[0].append(typs)
            multipoles[1].append(vals)
        
        self.multipole_rules = multipole_rules
        for line in bnd:
            typs = line.split()[1:3]
            val = [float(a) for a in line.split()[3:5]]
            bond[0].append(typs)
            bond[1].append(val[0])
            bond[2].append(val[1])
            
        for line in angl:
            s = line.split()
            typs = s[1:4]
            val = [float(a) for a in s[4:6]]
            angle[0].append(typs)
            angle[1].append(val[0])
            angle[2].append(val[1])

            if len(s) > 6:
                angle[3].append(s[6])
            else:
                angle[3].append("")


        for line in strb:
            typs = line.split()[1:4]
            val = [float(a) for a in line.split()[4:6]]
            strbnd[0].append(typs)
            strbnd[1].append(val[0])
            strbnd[2].append(val[1])
        
        for line in opbe:
            typs = line.split()[1:5]
            val = float(line.split()[5])
            opbend[0].append(typs)
            opbend[1].append(val)
        
        for line in tors:
            typs = line.split()[1:5]
            val = [float(a) for a in line.split()[5:]]
            torsion[0].append(typs)
            torsion[1].append(val)

        for line in bflx:
            typs = line.split()[1:3]
            val = float(line.split()[3])
            bndcflux[0].append(typs)
            bndcflux[1].append(val)
            
        for line in aflx:
            typs = line.split()[1:4]
            val = [float(a) for a in line.split()[4:]]
            angcflux[0].append(typs)
            angcflux[1].append(val)

        for line in exchp:
            val = [float(a) for a in line.split()[1:]]
            exchpol.append(val)
        
        if len(exchpol) > 0:
            exchpol = np.array(exchpol)
            exchpol = exchpol[np.argsort(exchpol[:,0])]

        prmdict = {'atom': atoms,
                'types': atmtyp,
                'typcls': typcls,
                'bond': bond,
                'angle': angle,
                'strbnd': strbnd,
                'opbend': opbend,
                'torsion': torsion,
                'chgpen': chgpen,
                'dispersion': dispersion,
                'repulsion': repulsion,
                'polarize': polarize,
                'bndcflux': bndcflux,
                'angcflux': angcflux,
                'chgtrn': chgtrn,
                'multipole': multipoles,
                'exchpol': np.asarray(exchpol)  }   

        self.prmdict = prmdict
        self.initprmdict = prmdict.copy()
        
    def build_prm_list(self, termfit=[]):
        if len(termfit) == 0:
            termfit = self.termfit
        prmdict = self.prmdict
        prmfit = []
        c = 0
        inds = []
        for term,vals in prmdict.items():
            if term == 'types':
                continue
            if term == 'bond' and "bond-force" in termfit:
                prmfit += [v for v in vals[1]]
                inds.append([c,c+len(vals[1])])
                c += len(vals[1])
            if term == 'bond' and "bond-value" in termfit:
                prmfit += vals[2]
                inds.append([c,c+len(vals[2])])
                c += len(vals[2])
            if term == 'angle' and "angle-force" in termfit:
                prmfit += [v for v in vals[1]]
                inds.append([c,c+len(vals[1])])
                c += len(vals[1])
            if term == 'angle' and "angle-value" in termfit:
                prmfit += vals[2]
                inds.append([c,c+len(vals[2])])
                c += len(vals[2])
            if term == 'chgpen' and term in termfit:
                prmfit += [v for v in vals[:,1]]
                inds.append([c,c+len(vals[:,1])])
                c += len(vals[:,1])
            if term == 'dispersion' and term in termfit:
                prmfit += [v for v in vals]
                inds.append([c,c+len(vals)])
                c += len(vals)
            if term == 'repulsion' and term in termfit: 
                tv = 0
                for line in vals:
                    prmfit += [v for v in line]
                    tv += len(line)
                inds.append([c,c+tv])
                c += tv
            if term == 'polarize' and term in termfit:
                prmfit += [v for v in vals[0]]
                inds.append([c,c+len(vals[0])])
                c += len(vals[0])
            if term == 'bndcflux' and term in termfit:
                prmfit += vals[1]
                inds.append([c,c+len(vals[1])])
                c += len(vals[1])
            if term == 'angcflux' and term in termfit:
                tv = 0
                for line in vals[1]:
                    prmfit += line
                    tv += len(line)

                inds.append([c,c+tv])
                c += tv
            
            if term == 'chgtrn' and term in termfit:
                tv = 0
                for i,line in enumerate(vals):
                    vals2 = [v for v in line if v != 0]

                    # print(prmfit,len(vals2),tv)
                    prmfit += vals2
                    tv += len(vals2)

                self.init_ct = vals.copy()
                inds.append([c,c+tv])
                c += tv
            
            multipole_rules = self.multipole_rules
            if term == 'multipole' and term in termfit:
                nonindep = {}
                for i,nm in enumerate(multipole_rules.keys()):
                    val = multipole_rules[nm]
                    if isinstance(val,float):
                        nonindep[i] = False
                    else:
                        nonindep[i] = True
                
                self.multipole_nonindep = nonindep
                tv = 0
                for i,line in enumerate(vals[1]):
                    if nonindep[i]:
                        vals2 = [v for v in line[1:] if v != 0]
                    else:
                        vals2 = [v for v in line if v != 0]
                    prmfit += vals2
                    tv += len(vals2)
                inds.append([c,c+tv])
                c += tv
        
        self.initial_params = prmfit
        self.termfit = termfit
        self.termind = inds

    def prmlist_to_dict(self,prmfit):
        termfit = self.termfit
        inds = self.termind
        prmdict = self.prmdict
        newprmdict = prmdict.copy()

        ntyps = len(prmdict['types'])

        
        for w,term in enumerate(termfit):
            prm = prmfit[inds[w][0]:inds[w][1]]
            if term == 'bond-force':
                newprmdict['bond'][1] = prm
            if term == 'bond-value':
                newprmdict['bond'][2] = prm
            if term == 'angle-force':
                newprmdict['angle'][1] = prm
            if term == 'angle-value':
                newprmdict['angle'][2] = prm
            if term == 'chgpen':
                newprmdict[term][:,1] = prm
            if term == 'dispersion':
                newprmdict[term] = prm
            if term == 'repulsion':
                newprmdict[term] = np.array(prm).reshape([ntyps,3])
            if term == 'polarize':
                newprmdict['polarize'][0] = prm
            if term == 'bndcflux':
                newprmdict['bndcflux'][1] = prm
            if term == 'angcflux':
                naf = int(len(prm)/4)
                newprmdict['angcflux'][1] = np.array(prm).reshape([naf,4])
            if term == 'chgtrn':
                z = 0
                for k,val in enumerate(newprmdict['chgtrn']):
                    for i,v in enumerate(val):
                        testv = self.init_ct[k][i]
                        if testv != 0:
                            newprmdict['chgtrn'][k][i] = prm[z]
                            z+=1
         
            if term == 'multipole':
                nonindep = self.multipole_nonindep

                z = 0
                for k,val in enumerate(newprmdict['multipole'][1]):
                    for i,v in enumerate(val):
                        if nonindep[k] and i == 0:
                            continue
                        if v != 0:
                            newprmdict['multipole'][1][k][i] = prm[z]
                            z += 1
                
                for i,pind in nonindep.items():
                    if not pind: 
                        nm = f"c{i+1}"
                        val = newprmdict['multipole'][1][i][0]
                        exec(f"{nm}={val}")
                for i,pind in nonindep.items():
                    if pind: 
                        nm = f"c{i+1}"
                        expr = self.multipole_rules[nm]
                        val = eval(expr)
                        newprmdict['multipole'][1][i][0] = val
        
        return newprmdict

    def make_key(self,prms=[],keytype="both"):
        termfit = self.termfit
        if not isinstance(prms, dict):
            if len(prms) == 0:
                prmdict = self.prmdict
            else:    
                prmdict = self.prmlist_to_dict(prms)
        else:
            prmdict = prms.copy()

        n = self.molnumber

        potdir = f"{self.basedir}/{n}/potential-test"
        poldir = f"{self.basedir}/{n}/mol-polarize"
        dimerdir = f"{self.basedir}/{n}/dimer"
        liqdir = f"{self.basedir}/{n}/liquid"

        keyliq = f"""parameters          {n}.prm
integrator respa

dcd-archive
tau-pressure      5.00
tau-temperature   1.0
barostat          langevin
volume-trial      5

digits            10
printout          500

a-axis            30.0
cutoff            7
neighbor-list
ewald
dewald

polarization      mutual               
polar-eps         1e-05                     
polar-predict     aspc
#########################
"""
        keygas = f"""parameters          {n}.prm
integrator        stochastic

dcd-archive
tau-temperature   0.1
volume-scale      molecular
THERMOSTAT        BUSSI
BAROSTAT          MONTECARLO

digits            10
printout          5000

polarization      mutual               
polar-eps         1e-06                     
#########################
"""       
        keyfile = ""
        typcls = prmdict['typcls']
        
        for k,t in enumerate(prmdict['types']):
            acls = typcls[t]
            term = "polarize"
            v = prmdict[term][0][k]
            c = "  ".join(prmdict[term][1][k])
            keyfile += f"{term:16s} {t:<11d}{v:10.6f}  {c}\n"

            term = "chgpen"
            v = prmdict[term][k][0]
            cp = prmdict[term][k][1]
            keyfile += f"{term:16s} {acls:<11d}{v:8.4f} {cp:11.6f}\n"

            term = "dispersion"
            v = prmdict[term][k]
            keyfile += f"{term:16s} {acls:<11d}{v:10.6f}{cp:10.6f}\n"

            term = "repulsion"
            v = prmdict[term][k]
            keyfile += f"{term:16s} {acls:<11d}{v[0]:10.6f}{v[1]:10.6f}{v[2]:10.6f}\n"

            term = "chgtrn"
            v = prmdict[term][k][0]
            c = prmdict[term][k][1]
            keyfile += f"{term:16s} {acls:<11d}{v:10.6f}{c:10.6f}\n"

        term = "bond"
        if term in prmdict.keys():
            for k,v in enumerate(prmdict[term][1]):
                c = "  ".join(prmdict[term][0][k])
                v2 = prmdict[term][2][k]
                keyfile += f"{term:12s}  {c:<15s}{v:10.6f} {v2:10.6f}\n"
        
        term = "angle"
        if term in prmdict.keys():
            for k,v in enumerate(prmdict[term][1]):
                c = "  ".join(prmdict[term][0][k])
                v2 = prmdict[term][2][k]
                keyfile += f"{term:12s}  {c:<15s}{v:10.6f} {v2:12.6f}\n"

        term = "bndcflux"
        for k,v in enumerate(prmdict[term][1]):
            c = "  ".join(prmdict[term][0][k])
            keyfile += f"{term:12s}  {c:<15s}{v:10.6f}\n"
        term = "angcflux"
        for k,v in enumerate(prmdict[term][1]):
            c = "  ".join(prmdict[term][0][k])
            keyfile += f"{term:12s}  {c:<15s}{v[0]:10.6f}{v[1]:10.6f}{v[2]:10.6f}{v[3]:10.6f}\n"

        term = "multipole"
        for k,v in enumerate(prmdict[term][1]):
            c = "  ".join(prmdict[term][0][k])
            keyfile += f"{term:12s} {c:<15s}{v[0]:10.6f}\n"
            keyfile += f"{' ':28s}{v[1]:10.6f}{v[2]:10.6f}{v[3]:10.6f}\n"
            keyfile += f"{' ':28s}{v[4]:10.6f}\n"
            keyfile += f"{' ':28s}{v[5]:10.6f}{v[6]:10.6f}\n"
            v9 = -(v[4]+v[6])
            keyfile += f"{' ':28s}{v[7]:10.6f}{v[8]:10.6f}{v9:10.6f}\n"
        keyfile += "\n"

        if keytype == "liquid":
            with open(f"{liqdir}/liquid.key",'w') as file:
                file.write(keyliq+keyfile)
        elif keytype == 'gas':
            with open(f"{potdir}/monomer.key",'w') as file:
                file.write(keygas+keyfile)
            copy_files(f"{potdir}/monomer.key", f"{liqdir}/gas.key")
            copy_files(f"{potdir}/monomer.key", f"{dimerdir}/tinker.key")
            copy_files(f"{potdir}/monomer.key", poldir)
        else:
            with open(f"{liqdir}/liquid.key",'w') as file:
                file.write(keyliq+keyfile)
            with open(f"{potdir}/monomer.key",'w') as file:
                file.write(keygas+keyfile)
            copy_files(f"{potdir}/monomer.key", f"{liqdir}/gas.key")
            copy_files(f"{potdir}/monomer.key", f"{dimerdir}/tinker.key")
            copy_files(f"{potdir}/monomer.key", poldir)


    def prepare_checkdata(self,filenm=""):
        n = self.molnumber
        if len(filenm) == 0 or filenm == None:
            return
        
        if os.path.isfile(filenm):
            fname = filenm 
        elif os.path.isfile(f"{self.basedir}/{n}/dumpdata/{filenm}"):
            fname = f"{self.basedir}/{n}/dumpdata/{filenm}"
        elif os.path.isfile(f"{self.basedir}/{n}/{filenm}"):
            fname = f"{self.basedir}/{n}/{filenm}"
        else:
            return
        
        self.usedatafile = True
        self.chkdata = load_pickle(fname)

    ## prepare dimer files
    def prepare_opt_sapt_dimers(self):
        n = self.molnumber
        qmpath = f"{self.datadir}/qm-calc/{n}"
        dimerdir = f"{self.basedir}/{n}/dimer"

        os.chdir(dimerdir)

        refnm_dimers = ["water+mol","mol+mol"]
        # refnm_dimers = ["water+mol"]
        nconf = 5

        nm_dimers = []
        ref_energy = {}
        for nm in refnm_dimers:
            if os.path.isfile(f"{qmpath}/sapt-res-{nm}.npy"):
                copy_files(f"{qmpath}/sapt-res-{nm}.npy",dimerdir)

                files = next(os.walk(qmpath))[2]
                for fn in files:
                    if 'conf' in fn and nm in fn and 'xyz' in fn:
                        copy_files(f"{qmpath}/{fn}",dimerdir)

                ref_energy[nm] = np.load(f"{qmpath}/sapt-res-{nm}.npy")
                nm_dimers.append(nm)
            if os.path.isfile(f"{qmpath}/ccsdt-{nm}.npy"):
                rtotal = np.load(f"{qmpath}/ccsdt-{nm}.npy")
                ref_energy[nm][:,-1] = rtotal

        self.nm_dimers = nm_dimers
        self.nconf = 5

        for nm in nm_dimers:
            for k in range(1,nconf+1):
                xyznm = f"{nm}-conf_{k}"
                fn = f"{qmpath}/{xyznm}.xyz"
            
                f = open(fn)
                dt = f.readlines()
                f.close()
            
                header_line = dt[0].strip('\n')

                ml = int(header_line.split()[0])
                nmol1 = self.Natoms
                nmol2 = ml - nmol1

                mol1 = [a.strip('\n') for a in dt[1:nmol1+1]]
                mol2 = [a.strip('\n') for a in dt[nmol1+1:]]
                
                hd_1 = [f"    {nmol1} Mol1"]
                hd_2 = [f"    {nmol2} Mol2"]
                np.savetxt(f'{xyznm}-mol1.xyz',hd_1+mol1,fmt="%s")
                np.savetxt(f'{xyznm}-mol2.xyz',hd_2+mol2,fmt="%s")
        
        self.ref_energy = ref_energy
        self.do_dimers = True
        os.chdir(self.basedir)

    def prepare_cluster(self):
        n = self.molnumber
        qmpath = f"{self.datadir}/qm-calc/{n}"
        dimerdir = f"{self.basedir}/{n}/dimer"

        if not os.path.isdir(f"{qmpath}/clusters"):
            self.do_clusters = False
            return
        
        res = load_pickle(f"{qmpath}/clusters/sapt-res-water-cluster.pickle")
        for nm,vals in res.items():
            copy_files(f"{qmpath}/clusters/{nm}.xyz",dimerdir)
        copy_files(f"{qmpath}/clusters/water.xyz",dimerdir)
        copy_files(f"{qmpath}/clusters/mol.xyz",dimerdir)

        self.cluster_ref = res
        self.cluster_names = list(res.keys())
        self.do_clusters = True

    def prepare_sapt_dimers(self):
        n = self.molnumber
        qmpath = f"{self.datadir}/qm-calc/{n}"
        dimerdir = f"{self.basedir}/{n}/dimer"

        if not os.path.isdir(f"{qmpath}/sapt_dimers"):
            return
        
        files = next(os.walk(f"{qmpath}/sapt_dimers"))[2]
        fnames = []
        for fn in files:
            if 'arc' == fn[-3:]:
                fnames.append(fn[:-4])
                copy_files(f"{qmpath}/sapt_dimers/{fn}",dimerdir)
        for sysn in fnames:
            for fn in files:
                if sysn in fn and 'xyz' in fn:
                    copy_files(f"{qmpath}/sapt_dimers/{fn}",dimerdir)

        self.sapt_dimers = fnames
        self.sapt_dimers_ref = {}
        self.sapt_dimers_indx = {}
        for fn in fnames:
            ref = np.load(f"{qmpath}/sapt_dimers/{fn}.npy")
            ndim = ref.shape[0]

            ref1 = ref[:,-1]
            cut = 2*np.std(ref1)
            if cut > 8:
                cut = 8
            elif (cut-ref1.max()) < 5:
                cut = ref1.max()+1
                if ref1.max() > 8:
                    cut = 8
            mask = ref1 < cut
            indx = np.arange(ndim)[mask]
            self.sapt_dimers_ref[fn] = ref[indx].copy()
            self.sapt_dimers_indx[fn] = indx.copy()

        # Exclude fitting to DESRES sapt data if you have other data
        include_DESRES = {}
        total_included = 0
        for nm in fnames:
            if 'DESRES' in nm:
                include_DESRES[nm] = False          
                if'water' not in nm:
                    include_DESRES[nm] = True
                    total_included += 1
                else:
                    if not self.do_dimers:
                        include_DESRES[nm] = True
                        total_included += 1
            else:
                total_included += 1

        self.sapt_dimers_include_DESRES = include_DESRES

        if total_included:
            self.do_sapt_dimers = True

    def prepare_ccsdt_dimers(self):
        n = self.molnumber
        qmpath = f"{self.datadir}/qm-calc/{n}"
        dimerdir = f"{self.basedir}/{n}/dimer"

        if not os.path.isdir(f"{qmpath}/ccsdt_dimers"):
            self.do_ccsdt_dimers = False
            return
        
        files = next(os.walk(f"{qmpath}/ccsdt_dimers"))[2]
        fnames = []
        for fn in files:
            if 'arc' == fn[-3:]:
                fnames.append(fn[:-4])
                copy_files(f"{qmpath}/ccsdt_dimers/{fn}",dimerdir)
        
        self.ccsdt_dimers = fnames
        self.ccsdt_dimers_ref = {}
        self.ccsdt_dimers_indx = {}
        self.ccsdt_dimers_weights = {}
        for fn in fnames:
            ref = np.load(f"{qmpath}/ccsdt_dimers/{fn}.npy")
            ndim = ref.shape[0]

            cut = 2*np.std(ref)
            if cut > 8:
                cut = 8
            elif (cut-ref.max()) < 5:
                cut = ref.max()+1
                if ref.max() > 8:
                    cut = 8
            mask = ref < cut
            indx = np.arange(ndim)[mask]
            self.ccsdt_dimers_ref[fn] = ref[indx].copy()
            self.ccsdt_dimers_indx[fn] = indx.copy()

            nvals = self.ccsdt_dimers_ref[fn].shape[0]
            self.ccsdt_dimers_weights[fn] = np.ones(nvals)
            if 'DESRES' not in fn:
                eref = self.ccsdt_dimers_ref[fn]
                emin = np.min(eref)
                ixx = np.where(emin==eref)[0][0]
                if ixx != 0 and ixx != nvals-1:
                    weig = np.zeros(nvals)
                    ns = int(ixx/2)
                    weig[ns:ixx+ns] += 2
                    weig[:ns] += 0.1
                    weig[ixx+ns:] += 0.5
                    self.ccsdt_dimers_weights[fn] = weig

        self.do_ccsdt_dimers = True
        os.chdir(self.basedir)
        
    def get_polarize(self):
        n = self.molnumber
        poldir = f"{self.basedir}/{n}/mol-polarize"
        os.chdir(poldir)

        os.system("rm -f *.err*")

        cmd1 = f"{tinkerpath}/bin/polarize monomer"
        out_log = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE,encoding='utf8')
        output = out_log.communicate()
        output = output[0].split('\n')


        os.chdir(self.basedir)

        line = output[-4].split()
        try:
            pol = np.array([float(a) for a in line])
        except:
            pol = np.array([1e4,1e4,1e4])

        ref = np.abs(self.refmolpol)
        rms = np.abs(pol-ref)
        
        d1 = np.abs(pol[0]-ref[2])
        d2 = np.abs(pol[2]-ref[0])
        
        if d1 + d2 < rms[0] + rms[2]:
            pol2 = np.array([pol[2],pol[1],pol[0]])
        else:
            pol2 = pol.copy()

        abspol = np.abs(pol2)
        rms = np.abs(abspol-ref)
        avgpol = np.abs(ref.mean() - abspol.mean())
        for ii,cmp in enumerate(ref):
            if np.abs(cmp) < 1e-5:
                if rms[ii] > 3:
                    rms[ii] = avgpol

        self.molpol = pol2.copy()        
        return rms
    
    def get_potfit(self):
        n = self.molnumber
        potdir = f"{self.basedir}/{n}/potential-test"
        os.chdir(potdir)

        os.system("rm -f *.err*")

        cmd1 = f"{tinkerpath}/bin/potential 5 monomer monomer n"
        out_log = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE,encoding='utf8')
        output = out_log.communicate()
        output = output[0].split('\n')


        os.chdir(self.basedir)

        try:
            rms = float(output[-2].split()[-1])
        except:
            rms = 100

        return rms
    
    def analyze(self,xyzfn,keyfile="tinker.key",opt='e'):
        n = self.molnumber
        dimerdir = f"{self.basedir}/{n}/dimer"
        os.chdir(dimerdir)

        cmd = f'{tinkerpath}/bin/analyze'

        if xyzfn[-4:] != '.xyz':
            xyzfn += '.xyz'
        
        if keyfile == 'tinker.key':
            inp = f"{cmd} {xyzfn} {opt}"
        else:
            inp = f"{cmd} {xyzfn} {opt} -k {keyfile}"
        
        result = subprocess.run(inp, stdout=subprocess.PIPE,shell=True)
        out = result.stdout.decode('utf-8')
        all_out = out.split('\n')
        all_out = np.array(all_out)
        all_out = all_out[all_out != '']
        all_out = all_out[-18:]
        intermol = 0
        ps = 0
        for p,l in enumerate(all_out):
            if 'Intermolecular Energy' in l:
                intermol = float(l.split()[-2])
                ps = p+3
                break

            elif 'Total Potential Energy' in l:
                total = float(l.split()[-2])
                ps = p+2
                break
        
        if ps == 0:
            eng_cpm = np.zeros(len(energy_terms),dtype=float)+100
            intermol = 1e6
            return eng_cpm,intermol
        split = [a.split() for a in all_out[ps:]]
        values = np.array([[a[-3],a[-2]] for a in split])

        eng_cpm = np.zeros(len(energy_terms),dtype=float)
        for nt,term in enumerate(energy_terms):
            ids = np.where(term==values[:,0])[0]
            if ids.shape[0] > 0:
                vals = float(values[:,1][ids])
                eng_cpm[nt] = vals
        os.chdir(self.basedir)
        return eng_cpm,intermol

    def analyze_arc(self,nm_dimers,keyfile="tinker.key",intermolecular=True):     
        n = self.molnumber
        dimerdir = f"{self.basedir}/{n}/dimer"
        os.chdir(dimerdir)

        os.system("rm -f *.err*")

        cmd = f'{tinkerpath}/bin/analyze'

        single = False
        fnames = nm_dimers
        if isinstance(nm_dimers,str) :
            fnames = [nm_dimers]
            single = True
        
        all_componts = []
        allinter = []
        for k,nm in enumerate(fnames):
            if os.path.isfile(f"{nm}.arc"):
                xyzfn = f"{nm}.arc"
            elif os.path.isfile(f"{nm}.xyz") and not os.path.isfile(f"{nm}.arc"):
                xyzfn = f"{nm}.xyz" 
            else:
                return
            
            inp = f"{cmd} {xyzfn} e -k {keyfile}"
            result = subprocess.run(inp, stdout=subprocess.PIPE,shell=True)
            out = result.stdout.decode('utf-8')
            all_out = out.split('\n')

            all_out = np.array(all_out)
            all_out = all_out[all_out != '']
            all_out = all_out[12:]

            frms = [p for p,a in enumerate(all_out) if 'Intermolecular Energy' in a]
            nointer = False
            if len(frms) == 0:
                frms = [p for p,a in enumerate(all_out) if 'Total Potential' in a]
                nointer = True
            nfrms = len(frms)
            nstart = frms[0]

            split = [a.split() for a in all_out[nstart:] if 'Analysis' not in a and 'Breakdown' not in a]

            if nointer:
                frms = [p for p,a in enumerate(split) if 'Total' == a[0]]
            else:
                frms = [p for p,a in enumerate(split) if 'Intermolecular' == a[0]]
            
            energies = {a[-3]:np.zeros(nfrms,dtype=float) for a in split if ':' not in a}
            if nointer:
                energies['Total'] = np.zeros(nfrms,dtype=float)
            else:
                energies['Total'] = np.zeros(nfrms,dtype=float)
                energies['Intermolecular'] = np.zeros(nfrms,dtype=float)
            
            frms2 = frms + [len(split)]
            for ii,kk in enumerate(frms2[:-1]):
                k1 = kk 
                k2 = frms2[ii+1]
                
                if nointer:
                    s = split[k1]
                    energies['Total'][ii] += float(s[-2])
                    st = 1
                else:
                    s = split[k1]
                    energies['Intermolecular'][ii] += float(s[-2])
                    s = split[k1+1]
                    energies['Total'][ii] += float(s[-2])
                    st = 2
                
                for s in split[k1+st:k2]:
                    energies[s[-3]][ii] += float(s[-2])
            
            eng_cpm = np.zeros((nfrms,len(energy_terms)),dtype=float)
            for nt,term in enumerate(energy_terms):
                if term in energies.keys():
                    eng_cpm[:,nt] = energies[term].copy()

            if nfrms == 1:        
                final_energy = eng_cpm[0]
            else:
                final_energy = eng_cpm
            
            all_componts.append(final_energy)
            allinter.append(energies['Intermolecular'])
        os.chdir(self.basedir)
        if intermolecular:
            if single:
                return np.array(allinter[0]),np.array(all_componts[0])
            else:
                return np.array(allinter),np.array(all_componts)
        else:
            if single:
                return np.array(all_componts[0])
            else:
                return np.array(all_componts)

    def compute_water_dimers(self):
        n = self.molnumber
        nm_dimers = self.nm_dimers
        dimerdir = f"{self.basedir}/{n}/dimer"
        qmpath = f"{self.datadir}/qm-calc/{n}"
        
        os.chdir(dimerdir)
        os.system("rm -f *.err*")

        cmd = f'{tinkerpath}/bin/analyze'
        
        all_componts = []
        errors = []

        for nm in nm_dimers:
            ref_energy = self.ref_energy[nm]
            for k in range(1,self.nconf+1):
                xyznm = f"{nm}-conf_{k}"
                        
                mols = [xyznm,f'{xyznm}-mol1',f'{xyznm}-mol2']
            
                comp_energ = []
                for mll in mols:
                    inp = f"{mll}.xyz\nge\n"
                    input_ = inp.encode('utf-8')
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, input=input_)
                    out = result.stdout.decode('utf-8')
                    all_out = out.split('\n')
                    bb = all_out[-11:]
                    
                    eng_cpm = []
                    eterm = []
                    for line in bb:
                        try:
                            l2 = line.split()
                            eng_cpm.append(float(l2[-2]))
                            eterm.append(l2[-3])
                        except:
                            None
                    
                    eterm = np.array(eterm)
                    sorted_terms = []
                    for term in energy_terms:
                        if term not in eterm:
                            sorted_terms.append(0.0)
                        else:
                            ind = np.where(term==eterm)[0][0]
                            sorted_terms.append(eng_cpm[ind])
                            
                    comp_energ.append(sorted_terms)
                    #if nm1 == mll:
                    #    print('main',np.array(sorted_terms))
                
                comp_energ = np.array(comp_energ)
            #print('mono',(comp_energ[1]+comp_energ[2]))
                final_energy = comp_energ[0]-(comp_energ[1]+comp_energ[2])
                        
                calc_components = [final_energy[9],final_energy[7],final_energy[10]+final_energy[11], final_energy[8],final_energy.sum()]
                ###################Electrostat,        Exchange   , ------Induction-----         ,   Dispersion    ,   TOTAL
                
                #print(nm,final_energy.sum(),np.array(calc_components[0:4]).sum())
                all_componts.append(calc_components.copy())
                ref = ref_energy[k-1]

                #### Log scale for repulsion energy
                # calc_components[1] = np.log(calc_components[1])
                # ref[1] = np.log(np.abs(ref[1]))
                ####

                
                err = (calc_components - ref)
                # err = (calc_components - ref_energy)/np.abs(ref_energy)
                # if np.abs(ref_energy[-1]) < 1:
                #     err[-1] = err[-1]*np.abs(ref_energy[-1])
                if ref[-1] < 10 and ref[-1] > -12:
                    errors.append(err)
            os.chdir(self.basedir)
        
        return np.array(all_componts),np.array(errors)
    
    def compute_sapt_dimers(self):  
        n = self.molnumber
        dimerdir = f"{self.basedir}/{n}/dimer"      
        os.chdir(dimerdir)

        nm_dimers = self.sapt_dimers
        ref_energy = self.sapt_dimers_ref

        all_componts = []
        errors = []
        for k,nm in enumerate(nm_dimers):
            fnames = [f"{nm}",f"{nm}-mol1.xyz",f"{nm}-mol2.xyz"]
                        
            comps = self.analyze_arc(fnames[0],'tinker.key',False)
            indx = self.sapt_dimers_indx[nm]
            ndim = int(indx.shape[0])
            if comps.sum() > 1e5:
                res1 = np.zeros((ndim,5))+100
                err = np.zeros((ndim,5))+1e6
            else:
                terms1 = np.array([comps[:,9],comps[:,7],comps[:,10]+comps[:,11], comps[:,8],comps.sum(axis=1)])
                terms1 = terms1.T
                comps,inter2 = self.analyze(fnames[1])
                terms2 = np.array([comps[9],comps[7],comps[10]+comps[11], comps[8],comps.sum()])
                comps,inter3 = self.analyze(fnames[2])
                terms3 = np.array([comps[9],comps[7],comps[10]+comps[11], comps[8],comps.sum()])

                final_energy = terms1-(terms2+terms3)
                res1 = final_energy[indx]
                ref = ref_energy[nm]
                
                err = (res1 - ref)
                if 'DESRES' in nm:
                    err[:,1] = np.zeros(ndim)
                    err[:,2] = np.zeros(ndim)
                    err[:,3] = np.zeros(ndim)

                    if self.sapt_dimers_include_DESRES[nm]:
                        errors.append(err)
                else:
                    errors.append(err)
            all_componts.append(res1)
        
        os.chdir(self.basedir)

        if len(nm_dimers) > 1:
            allerr = errors[0]
            for e in errors[1:]:
                allerr = np.concatenate((allerr,e))

            return all_componts,allerr
        else:
            return np.array(all_componts[0]),np.array(errors[0])

    def compute_cluster(self):
        n = self.molnumber
        dimerdir = f"{self.basedir}/{n}/dimer"      
        os.chdir(dimerdir)

        nm_dimers = self.cluster_names
        allref = self.cluster_ref

        termsw, intenw = self.analyze("water.xyz")
        termsb, intenb = self.analyze("mol.xyz")

        if intenw > 1e5 or intenb > 1e5:
            res = np.zeros((len(nm_dimers),5))+100
            err = np.zeros((len(nm_dimers),5))+1e6
            return res,err

        all_componts = []
        errors = []
        for k,nm in enumerate(nm_dimers):
            nw = int(nm[1])
            nb = int(nm[3])
                        
            fn = f"{nm}.xyz"
            terms, inten = self.analyze(fn)

            if inten > 1e5:
                res = np.zeros((len(nm_dimers),5))+100
                err = np.zeros((len(nm_dimers),5))+1e6
                return res,err
            
            final = terms - nw*termsw - nb*termsb
            
            comps = [final[9],final[7],final[10]+final[11], final[8],final.sum()]
            ref = allref[nm]

            err = (np.array(comps) - ref)
            errors.append(err)
            all_componts.append(comps)

        os.chdir(self.basedir)
        return np.array(all_componts),np.array(errors)

    def compute_ccsdt_dimer(self):
        n = self.molnumber
        dimerdir = f"{self.basedir}/{n}/dimer"      
        os.chdir(dimerdir)

        nm_dimers = self.ccsdt_dimers
        allref = self.ccsdt_dimers_ref

        all_componts = []
        errors = []

        inter_energy,all_comps = self.analyze_arc(nm_dimers)        
        for k,nm in enumerate(nm_dimers):
            comps = inter_energy[k]
            ref = allref[nm]

            indx = self.ccsdt_dimers_indx[nm]
            ndim = int(indx.shape[0])
            res1 = comps[indx]
            if res1.sum() > 1e5 or len(res1) != len(ref):
                res1 = np.zeros(ndim)+100
                err = np.zeros(ndim)+1e6

                errors.append(err)
                all_componts.append(res1)
            else:
                err = np.abs(res1 - ref)
            
                testerr = np.abs(err)
                testerr = np.sort(testerr)[::-1]
                ndim = int(ndim/2)

                if ndim > 10:
                    ndim = 10
                
                if 'DESRES' in nm:
                    err1 = testerr.mean()+testerr[:ndim].mean()
                else:
                    err *= self.ccsdt_dimers_weights[nm]
                    err1 = np.abs(err).sum()
                errors.append(err1)
                all_componts.append(res1)
        
        os.chdir(self.basedir)
        if len(nm_dimers) == 1:
            return res1,errors
        return np.array(all_componts),errors
    
    def liquid_fitproperties(self):
        n = self.molnumber

        try:
            info = self.molinfo[n]
        except:
            info = [273.15,800.0]
            liqprop = ["Temp", "Dens"]
            self.liquidref = [info,liqprop]
            return
        nsteps = self.nsteps
        simlen = (nsteps*2/1000)

        refvalues = info[:2]
        liqprop = ["Temp", "Dens"]
        if info[2] != -1:
            refvalues.append(info[2])
            liqprop.append("HV")
        
        # n Temp Dens HV Diel KT alphaT SurfTens
        if simlen >= 2000:
            if info[3] != -1:
                refvalues.append(info[3])
                liqprop.append("Diel")
        if simlen >= 2500:
            if info[4] != -1:
                refvalues.append(info[4])
                liqprop.append("KT")
            if info[5] != -1:
                refvalues.append(info[5])
                liqprop.append("alphaT")
        
        self.liquidref = [refvalues,liqprop]

    def checkparams(self,testprms):
        for n,vals in self.chkdata.items():
            prms = vals['params']

            if len(prms) != len(testprms):
                break

            test = np.abs(testprms - prms).sum()

            if test < 1e-5:
                return vals
            
        return {}

    def minimize_box(self,filenm='liquid.xyz',erase=True,path=None):
        n = self.molnumber
        if path != None:
            liqdir = path
        else:
            liqdir = f"{self.basedir}/{n}/liquid"

        os.chdir(liqdir)

        if erase:
            os.system(f"rm -rf *.xyz_* *.err* *.end")
        else:
            os.system(f"rm -rf *.err* *.end")
        
        if 'xyz' in filenm:
            xyz_file = filenm
        elif 'arc' not in filenm:
            xyz_file = filenm+'.xyz'

        if 'gas' in filenm:
            cmd = f'{self.tinkerpath}/minimize {xyz_file} 0.1'
        else:
            cmd = f'{self.tinkerpath}/tinker9 minimize {xyz_file} 0.1' 

        out_log = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf8', preexec_fn=os.setsid)
        rerun = False
        try:
            output = out_log.communicate(timeout=300)
            all_out = output[0].split('\n')
        except subprocess.TimeoutExpired:
            job_pid1 = os.getpgid(out_log.pid)
            os.killpg(os.getpgid(job_pid1), signal.SIGTERM)
            out_log.kill()
            output = out_log.communicate()
            all_out = output[0].split('\n')
            rerun = True

        bb = all_out[-4:-1]

        error = False
        rms = 100
        if "Final RMS" in bb[1]:
            line1 = bb[0].strip('\n')
            line1 = line1.replace('D','e')
            line2 = bb[1].strip('\n')
            line2 = line2.replace('D','e')

            rms = float(line2.split()[-1])
            min_energ = float(line1.split()[-1])

        ### Check for incomplete convergence in single precision
        if 'Incomplete Convergence' in all_out[-6] and rms > 0.2 and rms < 100:
            rerun = True
        if rerun:
            cmd = f'{self.tinkerpath}/tinker9-double minimize {xyz_file} 0.1' 

            out_log = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf8', preexec_fn=os.setsid)
            try:
                output = out_log.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                job_pid1 = os.getpgid(out_log.pid)
                os.killpg(os.getpgid(job_pid1), signal.SIGTERM)
                out_log.kill()
                output = out_log.communicate()
            
            all_out = output[0].split('\n')
            bb = all_out[-4:-1]
            
            if "Final RMS" in bb[1]:
                line1 = bb[0].strip('\n')
                line1 = line1.replace('D','e')
                line2 = bb[1].strip('\n')
                line2 = line2.replace('D','e')

                rms = float(line2.split()[-1])
                min_energ = float(line1.split()[-1])

        if rms == 100 or np.isnan(rms) or np.isinf(rms):
            min_energ = -1.1e6
            rms = 1e4
            error = True

        elif rms > 10 or np.abs(min_energ) > 1.5e5:
            error = True

        os.chdir(self.basedir)

        return error, min_energ, rms


    def calltinker(self,command, nsteps=None, elfn=None, cudad=0):

        """ Call TINKER; prepend the tinkerpath to calling the TINKER program. """

        if nsteps == None:
            nsteps = self.nsteps

        error = False
        if isinstance(command,(list,)):
            csplit = command[0].split()
            csplit2 = command[1].split()

        else:
            csplit = command.split()
            csplit2 = None

        currdir = os.getcwd()

        tinkerpath = f'{self.tinkerpath}'
        tinker9 = f"{tinkerpath}/tinker9"
        if csplit[0] == 'dynamic':

            prog = f"{tinker9} dynamic"
            csplit[0] = prog
            
            commandd = ' '.join(csplit) + ' > liquid.log 2>&1'
            if elfn != None:
                cmd_liq = f"ssh elf{elfn} 'cd {currdir} && {commandd}' "
            else:
                cmd_liq = commandd

            liqproc = subprocess.Popen("exec "+cmd_liq, shell=True, universal_newlines='expand_cr')
            job_pid1 = os.getpgid(liqproc.pid)
            filename = os.path.abspath("./liquid.log")

            rungas = False
            if csplit2 != None:
                prog = os.path.join(tinkerpath, csplit2[0])
                csplit2[0] = prog
                commd2 = ' '.join(csplit2) + ' > gas.log 2>&1'

                nsteps_gas = int(csplit2[2])
                dt_gas = float(csplit2[3])
                simtime_gas = (1e-6)*nsteps_gas*dt_gas
                gas_run = subprocess.Popen("exec "+commd2, shell=True, universal_newlines='expand_cr')
                rungas = True
            
            simlen = (nsteps*2/1000)
            init_time = time.time()

            if simlen < 100:
                sleeper = 40
            else:
                sleeper = 60
            
            if simlen < 50:
                timeout = 5*60
            else:
                timeout = 2.5*simlen*3.6
            last_frame = 0
            diff_timer = 0
            diff = 5

            time.sleep(sleeper)

            running = True
            sucess = False

            last_frame = get_last_frame(filename)
            if last_frame == simlen:
                running = False
                sucess = True
            
            if not sucess and simlen < 100:
                running = False
                time.sleep(int(sleeper/3))
                last_frame = get_last_frame(filename)
                if last_frame == simlen:
                    sucess = True

            while running:
                new_last_frame = get_last_frame(filename)
                diff = new_last_frame - last_frame

                if diff == 0: 
                    if new_last_frame == simlen:
                        sucess = True
                        running = False
                        break

                    if diff_timer != 0:
                        totaltime = time.time() - diff_timer
                        if totaltime > sleeper*2:
                            running = False
                            break                    

                    run_t = time.time() - init_time
                    if run_t > timeout:
                        running = False

                        if new_last_frame == simlen:
                            sucess = True
                        break

                    diff_timer = time.time()                  
                    time.sleep(sleeper)

                time.sleep(int(sleeper/2))
                last_frame = new_last_frame
            
            if sucess:
                liqproc.communicate()
            else:
                liqproc.kill()

            if rungas:
                filename = os.path.abspath("./gas.log")
                ngasfrm = get_last_frame(filename)

                init_time = time.time() 
                while ngasfrm < simtime_gas:
                    time.sleep(5)
                    diff_timer = time.time() - init_time
                    
                    if diff_timer > timeout:
                        gas_run.kill()
                    
                    time.sleep(5)
                    ngasfrm = get_last_frame(filename)
        
        else:
            if 'liquid' in csplit[1]:
                if 'analyze' in csplit[0]:
                    prog = f"{tinker9} analyze"
                if 'minimize' in csplit[0]:
                    prog = f"{tinker9} minimize"
                csplit[0] = prog

                cmd_ = ' '.join(csplit[:-2]) + ' >> ' + csplit[-1] + ' 2>&1'

                if elfn != None:
                    commd2 = f"ssh elf{elfn} 'cd {currdir} && {cmd_}' "
                else:
                    commd2 = cmd_
            
            else:
                prog = os.path.join(tinkerpath, csplit[0])
                csplit[0] = prog
                commd2 = ' '.join(csplit[:-2]) + ' > ' + csplit[-1] + ' 2>&1'

            _run = subprocess.Popen("exec " +commd2, shell=True, universal_newlines='expand_cr')
            _run.communicate()

            try:
                _run.kill()
            except:
                None
            
        return error


    def run_npt(self,equil=None, nsteps=None, nsteps_gas=None,elf_n=None):
        n = self.molnumber
        liqdir = f"{self.basedir}/{n}/liquid"
        refliq = f"{self.basedir}/{n}/ref_liquid"

        if nsteps == None:
            nsteps = self.nsteps
        if nsteps_gas == None:
            nsteps_gas = self.nsteps_gas
        if equil == None:
            equil = self.equil

        os.chdir(liqdir)

        os.system(f"rm -f *.dyn *.dcd *.arc *.err*")

        if self.useliqdyn:
            os.system(f"cp {refliq}/liquid.dyn . 2>/dev/null")
            os.system(f"cp {refliq}/gas.dyn . 2>/dev/null")

        
        info = self.liquidref[0]
        props = self.liquidref[1]
        temperature = info[0]

        cmd_liq = f"dynamic liquid {nsteps} 2 1 4 {temperature:.2f} 1.0 n"                                                                  
        cmd_gas = f"dynamic gas {nsteps_gas} 0.1 1 2 {temperature:.2f}"

        if self.rungas:
            res = self.calltinker([cmd_liq,cmd_gas], nsteps)
        else:
            os.system(f"cp {refliq}/gas.log . 2>/dev/null")
            res = self.calltinker(cmd_liq, nsteps)
            

        error = False
        if not os.path.isfile('liquid.log'):
            error = True
        
        os.system(f"{self.tinkerpath}/analyze liquid.xyz g > analysis.log")
        if os.path.isfile("liquid.dcd"):
            if nsteps > 100000:
                err = self.calltinker("analyze liquid.dcd liquid.xyz em >> analysis.log")
        else:
            error = True

        nframes = 0
        if os.path.isfile('liquid.log') and not error:
            cmd = "grep 'Frame Number' liquid.log | tail -1"
            out_log = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf8')
            output = out_log.communicate()
            all_out = output[0].split('\n')

            try:
                bb = all_out[0].split()
                nframes = int(bb[-1])
            except:
                None
            
            if nframes < 10:
                error = True
        else:
            error = True
        
        simlen = (nsteps*2/1000)
        simlen_gas = (nsteps_gas*0.1/1000)
        halfgas = int(simlen_gas/2)
        total = simlen-nframes

        nvals = len(info) - 1
        err = np.zeros(nvals)+1e6
        res = np.zeros(nvals)-100
        if not error:
            ngas = get_last_frame(f"{liqdir}/gas.log")
            if ngas > 1:
                gaslog='gas.log'
            
            if len(self.gasdcd) > 0 and not self.rungas:
                gasdcd = self.gasdcd
                gasxyz = self.gasdcd[:-4]+'.xyz'
                cmd = f"analyze {gasdcd} {gasxyz} e -k gas.key > gas2.log"
                self.calltinker(cmd)

                ngas2 = get_last_frame(f"{liqdir}/gas2.log")
                if ngas2 > 100:
                    gaslog = 'gas2.log'

            liquid = liqAnalyze.Liquid(liqdir,'liquid.xyz',self.Natoms,temperature,equil,
                            logfile='liquid.log',analyzelog='analysis.log',gaslog=gaslog,molpol=self.molpol.mean())

            if nsteps > 100000:
                liquid.get_dielectric('analysis.log',molpol=self.molpol.mean())
            if not liquid.error:
                Rho_avg = 1000*liquid.avgRho
                Hvap_avg = 4.184*liquid.HV
                Eps0 = liquid.dielectric
                kappa = liquid.kappa/100
                alphaT = (1e3)*liquid.alpha
                res = np.array([Rho_avg,Hvap_avg,Eps0,kappa,alphaT])

                total += 1

                if np.isnan(Hvap_avg) or np.isnan(Rho_avg) or np.isinf(Hvap_avg) or np.isinf(Rho_avg) or np.isnan(Eps0) or np.isinf(Eps0):
                    err = np.zeros(nvals)+1e6
                    res = np.zeros(nvals)-100
                    error = True
                elif np.isnan(kappa) or np.isnan(alphaT) or np.isinf(kappa) or np.isinf(alphaT):
                    err = np.zeros(nvals)+1e6
                    res = np.zeros(nvals)-100
                    error = True
                else:
                    ref1 = info[1]
                    err = np.array([100*(Rho_avg-ref1)/ref1])

                    c = 1
                    if 'HV' in props:
                        c+=1
                        ref2 = info[c]
                        e = 100*(Hvap_avg-ref2)/ref2
                        err = np.append(err,e)
                    if "Diel" in props:
                        c+=1
                        ref3 = info[c]
                        e = 100*(Eps0-ref3)/ref3
                        err = np.append(err,e)
                    if "KT" in props:
                        c+=1
                        ref4 = info[c]
                        e = 100*(kappa-ref4)/ref4
                        err = np.append(err,e)
                    if "alphaT" in props:
                        c+=1
                        ref5 = info[c]
                        e = 100*(alphaT-ref5)/ref5
                        err = np.append(err,e)
        
        os.chdir(self.basedir)

        return total*err,res
    

    def optimize_prms(self,new_params):
        global i
                
        optimizer = self.optimizer
        n = self.molnumber
        path_mol = f"{self.basedir}/{n}"
        termfit = self.termfit

        if self.testliq or self.fitliq:
            info = self.liquidref[0]
            nvals = len(info) - 1

        i+=1

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        print(st)
        print("iter #%d" % i)
        sys.stdout.flush()

        if self.usedatafile:
            test = self.checkparams(new_params)
            
            if len(test) > 1:
                allres = self.checkparams(new_params)
                totalerror = allres['error']

                #### For differential evolution
                if optimizer == 'genetic':
                    errors = totalerror.sum()
                else:
                    errors = totalerror.copy()

                if i > 1:
                    perror = self.log[i-1]['error']
                    if perror.shape[0] == totalerror.shape[0] or optimizer == 'genetic':
                        self.log[i] = allres
                        save_pickle(self.log, f"{self.dumpfile}_temp")

                        return errors
                else:  
                    self.log[i] = allres
                    save_pickle(self.log, f"{self.dumpfile}_temp")
                    return errors      

        allres = {'params': new_params,
                  'potrms': 0,
                  'dimers': [],
                  'ccsdt_dimers': [],
                  'clusters': [],
                  'sapt_dimers': [],
                  'polarize': [],
                  'liqres': [],
                  'error': [] }

        dumpres = {'params': new_params,
                  'potrms': 0,
                  'dimers': [],
                  'ccsdt_dimers': [],
                  'clusters': [],
                  'sapt_dimers': [],
                  'polarize': [],
                  'liqres': [],
                  'error': [] }

        os.chdir(path_mol)
        self.make_key(new_params,keytype="both")
        rms = self.get_potfit()

        err_pol = self.get_polarize()
        ## potential fit
        poterror = 0.0
        if 'chgpen' in termfit or 'multipole' in termfit:
            refrms = self.initpotrms

            # if rms > refrms+0.3:
            #     poterror = rms*(1e4)
            # else:
            poterror = rms*10
            allres['potrms'] = rms
            dumpres['potrms'] = rms
        
        sys.stdout.flush()
        errlist = []
        if self.do_dimers or self.computeall:
            calc_components, errors = self.compute_water_dimers()
            if 'chgpen' in termfit or 'multipole' in termfit:
                err = np.abs(errors)[:,0].sum()
                errlist.append(err)
            if 'dispersion' in termfit:
                err = np.abs(errors)[:,3].sum()
                errlist.append(err)
            if 'repulsion' in termfit:
                err = np.abs(errors)[:,1].sum()
                errlist.append(err)
            if 'polarize' in termfit or 'chgtrn' in termfit:
                err = np.abs(errors)[:,2].sum()
                errlist.append(err)
            if len(termfit) > 2:
                err = np.abs(errors)[:,4].sum()
                errlist.append(err)

            allres['dimers'] = [calc_components, errors]
            dumpres['dimers'] = [calc_components, errors]

        sys.stdout.flush()
        if self.do_ccsdt_dimers or self.computeall:
            calc_components, errors = self.compute_ccsdt_dimer()

            allres['ccsdt_dimers'] = [calc_components, errors]
            dumpres['ccsdt_dimers'] = errors

            errors = [5*a for a in errors]
            errlist += errors
        
        sys.stdout.flush()
        if self.do_clusters or self.computeall:
            calc_components, errors = self.compute_cluster()
            errloc = []        
            if 'chgpen' in termfit or 'multipole' in termfit:
                err = np.abs(errors)[:,0].sum()
                errlist.append(err)
                errloc.append(err)
            if 'dispersion' in termfit:
                err = np.abs(errors)[:,3].sum()
                errlist.append(err)
                errloc.append(err)
            if 'repulsion' in termfit:
                err = np.abs(errors)[:,1].sum()
                errlist.append(err)
                errloc.append(err)
            if 'polarize' in termfit or 'chgtrn' in termfit:
                err = np.abs(errors)[:,2].sum()
                errlist.append(err)
                errloc.append(err)

            if len(termfit) > 2:
                err = np.abs(errors)[:,4].sum()
                errlist.append(err)
                errloc.append(err)

            allres['clusters'] = [calc_components, errors]
            dumpres['clusters'] = errloc

        sys.stdout.flush()
        if self.do_sapt_dimers or self.computeall:
            calc_components, errors = self.compute_sapt_dimers()   
            ndim = int(errors.shape[0]/2)
            if ndim > 10:
                ndim = 10 
            
            errloc = [] 
            if 'chgpen' in termfit or 'multipole' in termfit:
                testerr = np.abs(errors)[:,0]
                testerr = np.sort(testerr)[::-1]
                # err = testerr.mean()+testerr[:ndim].mean()
                err = testerr.sum()
                errlist.append(err)
                errloc.append(err)
            if 'dispersion' in termfit:
                testerr = np.abs(errors)[:,3]
                testerr = np.sort(testerr)[::-1]
                # err = testerr.mean()+testerr[:ndim].mean()
                err = testerr.sum()
                errlist.append(err)
                errloc.append(err)
            if 'repulsion' in termfit:
                testerr = np.abs(errors)[:,1]
                testerr = np.sort(testerr)[::-1]
                err = testerr.mean()+testerr[:ndim].mean()
                errlist.append(err)
                errloc.append(err)
            if 'polarize' in termfit or 'chgtrn' in termfit:
                testerr = np.abs(errors)[:,2]
                testerr = np.sort(testerr)[::-1]
                # err = testerr.mean()+testerr[:ndim].mean()
                err = testerr.sum()
                errlist.append(err)
                errloc.append(err)

            if len(termfit) > 2:
                testerr = np.abs(errors)[:,4]
                testerr = np.sort(testerr)[::-1]
                err = testerr.mean()+testerr[:ndim].mean()
                errlist.append(err)
                errloc.append(err)

            allres['sapt_dimers'] = [calc_components, errors]
            dumpres['sapt_dimers'] = errloc

        sys.stdout.flush()
        poltest = 'chgpen' in termfit or 'multipole' in termfit or 'polarize' in termfit
        if poltest or self.computeall:
            allres['polarize'] = err_pol
            dumpres['polarize'] = err_pol

            if'polarize' in termfit and len(termfit) < 3:
                err_pol *= 4

            errlist += [a for a in err_pol]

        os.chdir(path_mol)
        
        sys.stdout.flush()
        if poterror > 0:
            errlist.append(poterror)

        totalerror = np.array(errlist).flatten()
        ### remove inf and nan
        for k, a in enumerate(totalerror):
            if np.isnan(a) or np.isinf(a):
                totalerror[k] = 1e3
        
        ### Minimize liquid box
        minbox = False
        if totalerror.mean() < 100 and optimizer == 'genetic':
            minbox = True
        elif optimizer != 'genetic':
            minbox = True
        # if len(termfit) > 1:
        if minbox:
            err, min_en, rms = self.minimize_box()

            if err:
                if min_en == -1.1e6:
                    r = np.random.rand()
                    # boxerr = (1+r)*(1e6)
                    boxerr = 1e6
                else:
                    boxerr = np.abs(rms)
            else:
                if rms <= 0.2:
                    boxerr = 0.001
                else:
                    boxerr = rms
            totalerror = np.append(totalerror,boxerr)
        else:
            # r = np.random.rand()
            boxerr = 1e6
            rms = 100
            totalerror = np.append(totalerror,(1e6))

        if self.testliq and not self.fitliq:
            if totalerror.sum() > 300:
                proxyerr = totalerror[:8].sum()
                totalerror = np.append(totalerror,proxyerr/5)
                
                err = np.zeros(nvals)+1e6
                res = np.zeros(nvals)-100
            else:
                if minbox and rms < 1 and boxerr < 10:
                    err,res = self.run_npt(5, 5000, 100000)
                    err = np.abs(err)
                else:
                    err = np.zeros(nvals)+1e6
                    res = np.zeros(nvals)-100    
                
                if res[0] != -100:
                    totalerror = np.append(totalerror,1e-3)
                else:
                    totalerror = np.append(totalerror,1e6)

            allres['liqres'] = [res, err]
            dumpres['liqres'] = [res, err]
        
        sys.stdout.flush()
        if self.fitliq:
            if rms < 1 and boxerr < 10:
                gaserr = self.minimize_box('gas.xyz',False)
                err,res = self.run_npt()
                err = np.abs(err)
            else:
                err = np.zeros(nvals)+1e6
                res = np.zeros(nvals)-100

            totalerror = np.append(totalerror,100*err)
            
            allres['liqres'] = [res, err]
            dumpres['liqres'] = [res, err]
        
        allres['error'] = totalerror
        dumpres['error'] = totalerror

        self.log[i] = dumpres

        save_pickle(self.log, f"{self.dumpfile}_temp")
        save_pickle(allres, self.dumpresult)           
        
        #### For differential evolution
        if optimizer == 'genetic':
            errors = totalerror.sum()
        else:
            errors = totalerror.copy()

        return errors

    def fit_data(self,optimizer='genetic',fitliq=False,testliq=False):
        os.chdir(self.basedir)

        n = self.molnumber
        path_mol = f"{self.basedir}/{n}"
        initprms = self.initial_params
        self.optimizer = optimizer

        self.fitliq = fitliq
        self.testliq = testliq
        if fitliq:
            self.liquid_fitproperties()
        
        self.log = {}
        bounds = []
        fail = False

        if optimizer == 'genetic':
            
            for brm in initprms:
                # if brm == 0:
                #     bounds.append((-1e-3,1e-3))
                    
                if brm > 10 and brm < 40:
                    bounds.append((round1(brm*0.70),round1(brm*1.30)))
                
                elif brm > 40:
                    bounds.append((round1(brm*0.95),round1(brm*1.05)))
                    
                elif brm < 0.0:
                    bounds.append((round1(brm*1.30),round1(-brm*0.7)))
                
                elif brm > 0 and brm < 0.4:
                    bounds.append((round1(-brm*0.7),round1(brm*1.30)))
                else:
                    bounds.append((round1(brm*0.80),round1(brm*1.20)))

            try:
                opt = optimize.differential_evolution(self.optimize_prms,bounds)
            except:
                fail = True
        else:
            ## make bounds
            ubounds = []
            lbounds = []
            for k,brm in enumerate(initprms):
                if brm < 0:
                    lbounds.append(2*brm)
                    ubounds.append(-2*brm)
                elif brm > 0 and brm < 2:
                    lbounds.append(0.7*brm)
                    ubounds.append(1.3*brm)
                else:
                    lbounds.append(0.5*brm)
                    ubounds.append(1.5*brm)

            ## make sure charges add to zero

            lbounds = np.array(lbounds)
            ubounds = np.array(ubounds)

            try:
                opt = optimize.least_squares(self.optimize_prms,initprms,
                jac='3-point',bounds=(lbounds,ubounds),f_scale=0.5,
                diff_step=0.01,loss='soft_l1',verbose=2)

                print(opt.x)
                sys.stdout.flush()

                ### Error at solution
                err = opt.fun
                final_prms = opt.x
                errors = self.optimize_prms(final_prms)
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                fail = True

        
        ### Save permanent dumpfile
        if os.path.isfile(self.dumpfile):
            alllog = load_pickle(self.dumpfile)

            if os.path.isfile(f"{self.dumpfile}_temp"):
                log = load_pickle(f"{self.dumpfile}_temp")

                alllog = {**alllog, **log}

                save_pickle(alllog, self.dumpfile) 
                os.system(f"rm {self.dumpfile}_temp")
        else:
            os.system(f"cp {self.dumpfile}_temp {self.dumpfile}")

        os.chdir(self.basedir)

        if fail:
            os.system(f"touch {path_mol}/FIT_ERROR")
            return 0
        else:
            os.system(f"touch {path_mol}/FIT_DONE")
        
        return opt

    