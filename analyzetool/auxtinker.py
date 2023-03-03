import os, sys
import numpy as np
import subprocess

global tinkerpath
tinkerpath = "~/tinker"

energy_terms = np.array(['Stretching', 'Bending', 'Stretch-Bend', 'Bend', 'Angle',
       'Torsion', 'Waals',
       'Repulsion', 'Dispersion', 'Multipoles', 'Polarization',
       'Transfer'], dtype='<U12')


def set_tinkerpath(path):
    global tinkerpath

    tinkerpath = path
    

def analyze(workdir,xyzfn,keyfile="tinker.key",opt='e',tkpath=""):
    currdir = os.getcwd()
    os.chdir(workdir)

    if len(tkpath) > 0:
        set_tinkerpath(tkpath)


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
    
    split = [a.split() for a in all_out[ps:]]
    values = np.array([[a[-3],a[-2]] for a in split])

    eng_cpm = np.zeros(len(energy_terms),dtype=float)
    for nt,term in enumerate(energy_terms):
        ids = np.where(term==values[:,0])[0]
        if ids.shape[0] > 0:
            vals = float(values[:,1][ids])
            eng_cpm[nt] = vals
    
    os.chdir(currdir)
    return eng_cpm,intermol

def analyze_arc(workdir,nm_dimers,keyfile="tinker.key",intermolecular=True,tkpath=""): 
    currdir = os.getcwd()    
    os.chdir(workdir)

    if len(tkpath) > 0:
        set_tinkerpath(tkpath)


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
        if len(frms) == 0:
            frms = [p for p,a in enumerate(all_out) if 'Total Potential' in a]
        
        nfrms = len(frms)
        nstart = frms[0]

        split = [a.split() for a in all_out[nstart:] if 'Analysis' not in a and 'Breakdown' not in a]
        values = np.array([[a[-3],a[-2]] for a in split if ':' not in a])

        interlist = []
        if intermolecular:
            intermol = np.array([float(a[-2]) for a in split if 'Intermolecular' in a])
            total = np.array([float(a.split()[-2]) for a in all_out[frms]])
            interlist.append(intermol)
        
        eng_cpm = np.zeros((nfrms,len(energy_terms)),dtype=float)
        for nt,term in enumerate(energy_terms):
            ids = np.where(term==values[:,0])[0]
            if ids.shape[0] > 0:
                vals = np.asarray(values[:,1][ids],dtype=float)
                eng_cpm[:,nt] = vals.copy()

        if nfrms == 1:        
            final_energy = eng_cpm[0]
        else:
            final_energy = eng_cpm
        
        all_componts.append(final_energy)
        allinter.append(interlist)
    
    os.chdir(currdir)
    if intermolecular:
        if single:
            return np.array(allinter[0]),np.array(all_componts[0])
        else:
            return allinter,all_componts
    else:
        if single:
            return np.array(all_componts[0])
        else:
            return all_componts
        
def sapt_components(final):
    final = np.asanyarray(final)

    ndim = final.ndim

    if ndim == 1:
        comps = [final[9],final[7],final[10]+final[11], final[8],final.sum()]
    else:
        comps = np.array([final[:,9],final[:,7],final[:,10]+final[:,11], final[:,8],final.sum(axis=1)])

    return comps