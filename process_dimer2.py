import numpy as np
import os
import sys
import subprocess

# base_dir = "/work/roseane/HIPPO/virial2_data/trial18-current/dimer_files/calculations-2"
base_dir = "/work/roseane/HIPPO/virial2_data/trial18-current/sim_298"
tkpath = "/user/roseane/tinker/bin/"

def run_analyze_arc(xyzfn):

    cmd1 = f"{tkpath}analyze {xyzfn} e > temp_energies.log 2>&1"
    out_log = subprocess.Popen(cmd1,shell=True,encoding='utf8')
    output = out_log.communicate()

    cmd2 = f"grep 'Intermolecular Energy' temp_energies.log > inter_energies.log 2>&1" 
    out_log = subprocess.Popen(cmd2,shell=True,encoding='utf8')
    output = out_log.communicate()

    energy = 0
    
    f = open(f'inter_energies.log')
    dt = f.readlines()
    f.close()

    output = [lin.strip('\n') for lin in dt]
    del dt

    all_frames = []
    for line in output:
        s = line.split()
        energy = float(s[-2].replace('D','e'))
        all_frames.append(energy)
    
    return np.array(all_frames),len(all_frames)

def run_analyze_arc2(xyzfn):

    cmd1 = f"{tkpath}testgrad {xyzfn} y n n"
    out_log = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE,encoding='utf8')
    output = out_log.communicate()
    data = output[0].split("\n")

    begin_line = np.array([a[1:16] for a in data])
    inds = np.where(begin_line == "Total Potential")[0]

    totproc = np.where(begin_line == "Analysis for Ar")[0]
    
    big_forces = []
    energies = []
    forces = []
    allen = []
    
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

            if len(ff) == 6:
                allen.append(e)

                if e > -15.0:
                    forces.append(ff)
                    energies.append(e)
                else:
                    big_forces.append(q)
    
    if totproc.shape[0] == 0:
        slen = 1
    else:
        slen = int(data[totproc[-1]].split()[-1])
    
    return energies, forces, big_forces, slen-1

def run_analyze_arc3(xyzfn):

    cmd1 = f"{tkpath}testgrad {xyzfn} y n n > energies+forces.log 2>&1"
    out_log = subprocess.Popen(cmd1,shell=True, encoding='utf8')
    output = out_log.communicate()

    cmd2 = f"grep 'Total Potential Energy' energies+forces.log > energies.log" 
    out_log = subprocess.Popen(cmd2,shell=True,encoding='utf8')
    output = out_log.communicate()

    cmd2 = f"grep -A 7 'Type      Atom' energies+forces.log > forces.log" 
    out_log = subprocess.Popen(cmd2,shell=True, encoding='utf8')
    output = out_log.communicate()

def proc_dimer(elfn,ind0,rdist):
    direc = f"{base_dir}/{rdist:.2f}_proc"
    os.chdir(f"{direc}")

    f = open(f'{direc}/dimer{elfn}tot.xyz')
    dt = f.readlines()
    f.close()

    alldimers = [lin.strip('\n') for lin in dt]
    del dt
    alldimers = np.array(alldimers)
    total = len(alldimers)/7
    alldimers = np.reshape(alldimers,(int(total),7))
    
    energies = []
    bad_dimers = []
    bad_dimers2 = []

    gsize = alldimers.shape[0]
    
    dim = alldimers.copy()
    en, slen = run_analyze_arc(f'{direc}/dimer{elfn}tot.xyz')

    if slen == gsize:
        finalize = 1
    else:
        finalize = 0

        j1 = ind0+slen
        j2 = slen

        bad_dimers.append(j1)
        bad_dimers2.append(j2)
        lsize = gsize-slen

        dim = dim[slen+1:]

    energies+=en
        
    while finalize == 0:
        os.system(f"rm -rf dimer{elfn}.err* dimer{elfn}tot.err*")
        np.savetxt(f'dimer{elfn}.xyz',dim,fmt="%s",delimiter='\n')
        en, slen = run_analyze_arc(f'{direc}/dimer{elfn}.xyz')

        if len(en) >= 1:
            energies+=en
        
        if slen >= dim.shape[0]:
           finalize = 1

        else:
            finalize = 0
            dim = dim[slen+1:]
            lsize -= slen

            j1 += slen
            j2 += slen
            bad_dimers.append(j1)
            bad_dimers2.append(j2)

        if len(bad_dimers)+len(energies) == gsize:
            finalize = 1

    energies = np.array(energies)
    
    if energies.shape[0] > 1:
        np.save(f"energies{elfn}.npy",energies.flatten())
    if len(bad_dimers) > 0:
        np.save(f"bad_dimers{elfn}.npy",np.array(bad_dimers))

        good_dimer = np.delete(alldimers,np.array(bad_dimers2),axis=0)
        np.savetxt(f'dimer{elfn}_final.arc',good_dimer,fmt="%s",delimiter='\n')

    return energies.flatten(), np.array(bad_dimers)

def proc_dimer2(elfn,ind0,rdist):
    direc = f"{base_dir}/{rdist:.2f}_calc"
    os.chdir(f"{direc}")

    f = open(f'{direc}/dimer{elfn}tot.xyz')
    dt = f.readlines()
    f.close()

    alldimers = [lin.strip('\n') for lin in dt]
    del dt
    alldimers = np.array(alldimers)
    total = len(alldimers)/7
    alldimers = np.reshape(alldimers,(int(total),7))
    gsize = alldimers.shape[0]

    en, ff, big_forces, slen = run_analyze_arc2(f'{direc}/dimer{elfn}tot.xyz')
    os.system(f"rm -rf dimer{elfn}tot.err* dimer{elfn}.err*")

    energies = []
    forces = []
    bad_dimers = []

    if slen+1 == gsize:
        finalize = 1

    else:
        finalize = 0

    if len(big_forces) > 0:
        for a in big_forces:
            bad_dimers.append((a))

    if len(en) >= 1:
        energies+=en
        forces += ff

    if finalize == 0:
        j1 = slen
        bad_dimers.append(j1)

        dim = alldimers[slen+1:]

        print(f"slen: {slen}, exc: {j1}, total: {gsize}")
        sys.stdout.flush()

        if j1 == gsize - 1:
            finalize = 1

    while finalize == 0:
        np.savetxt(f'dimer{elfn}.xyz',dim,fmt="%s",delimiter='\n')

        en, ff, big_forces, slen = run_analyze_arc2(f'{direc}/dimer{elfn}.xyz')

        os.system(f"rm -rf dimer{elfn}.err*")

        if len(en) >= 1:
            energies+=en
            forces += ff

        if len(big_forces) > 0:
            for a in big_forces:
                bad_dimers.append((a+j1+1))

        if dim.shape[0] == 1 or dim.shape[0] == 0:
            finalize = 1
        elif slen+1 == dim.shape[0]:
            finalize = 1
        else:
            try: 
                dim = dim[slen+1:]
                stotal = dim.shape[0]

                j1 = gsize - stotal - 1

                if j1 >= 0 and j1 < gsize:
                    bad_dimers.append(j1)
                    finalize = 0
                    print(f"slen: {slen}, exc: {j1}")
                    sys.stdout.flush()

                else:
                    finalize = 1
            except:
                finalize = 1        

        

    energies = np.array(energies)
    forces = np.array(forces)
    bad_dimers = np.array(bad_dimers)

    if energies.shape[0] > 1:
        np.save(f"energies{elfn}.npy",energies.flatten())
        np.save(f"forces{elfn}.npy",forces)
    if len(bad_dimers) > 0:
        # good_dimer = np.delete(alldimers,bad_dimers,axis=0)
        # np.savetxt(f'dimer{elfn}_final.arc',good_dimer,fmt="%s",delimiter='\n')
        
        bad_dimers += ind0
        np.save(f"bad_dimers{elfn}.npy",np.array(bad_dimers))

def proc_dimer3(elfn,ind0,rdist):
    direc = f"{base_dir}/{rdist:.2f}_calc"
    os.chdir(f"{direc}")

    # run_analyze_arc3(f'{direc}/dimer.arc')
    # os.system(f"rm -rf dimers.err* energies+forces.log")
    energies, slen = run_analyze_arc(f'{direc}/dimer.arc')

    if slen == 17635800:
        os.system(f"rm -rf temp_energies.log inter_energies.log")

    np.save(f"inter_energies.npy",energies.flatten())


def main():
    elfn = int(sys.argv[1])
    ind0 = int(sys.argv[2])
    rdist = float(sys.argv[3])

    os.chdir(base_dir)

    proc_dimer3(elfn,ind0,rdist)

if __name__ == "__main__":
    main()