import os, sys
import numpy as np
import subprocess
import signal

global tinkerpath
tinkerpath = "~/tinker"

energy_terms = np.array(['Stretching', 'Bending', 'Stretch-Bend', 'Bend', 'Angle',
       'Torsion', 'Waals',
       'Repulsion', 'Dispersion', 'Multipoles', 'Polarization',
       'Transfer'], dtype='<U12')


def set_tinkerpath(path):
    global tinkerpath

    tinkerpath = path
    

def analyze(workdir,xyzfn,keyfile="tinker.key",opt='e',tkpath="",tinker9=True):
    currdir = os.getcwd()
    os.chdir(workdir)

    if len(tkpath) > 0:
        set_tinkerpath(tkpath)

    if tinker9:
        cmd = f'{tinkerpath}/bin/tinker9 analyze'
    else:
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

def analyze_arc(workdir,nm_dimers,keyfile="tinker.key",intermolecular=True,tkpath="",tinker9=True): 
    currdir = os.getcwd()    
    os.chdir(workdir)

    if len(tkpath) > 0:
        set_tinkerpath(tkpath)


    os.system("rm -f *.err*")

    if tinker9:
        cmd = f'{tinkerpath}/bin/tinker9 analyze'
    else:
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


def minimize_box(path,filenm='liquid.xyz',rms=0.1,erase=True,keyfile="",tkpath=""):
    currdir = os.getcwd()
    os.chdir(path)

    if len(tkpath) > 0:
        set_tinkerpath(tkpath)
    
    if erase:
        os.system(f"rm -rf *.xyz_* *.err* *.end")
    else:
        os.system(f"rm -rf *.err* *.end")
    
    if 'xyz' in filenm:
        xyz_file = filenm
    elif 'arc' not in filenm:
        xyz_file = filenm+'.xyz'

    if 'gas' in filenm:
        cmd = f'{tinkerpath}/bin/minimize {xyz_file} {rms}'
    else:
        cmd = f'{tinkerpath}/bin/tinker9 minimize {xyz_file} {rms}' 

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
        if 'gas' not in filenm:
            rerun = True
    if rerun:
        cmd = f'{tinkerpath}/bin/tinker9-double minimize {xyz_file} 0.1' 

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

    os.chdir(currdir)

    return error, min_energ, rms