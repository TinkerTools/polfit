import numpy as np
import os
import sys
import subprocess

direc = "/work/roseane/HIPPO/virial2_data/trial18-current/dimer_files/2.00_proc"
tkpath = "/user/roseane/tinker/bin/"

def run_analyze_arc(xyzfn):

    cmd1 = f"{tkpath}analyze {xyzfn} e"
    out_log = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE,encoding='utf8')
    output = out_log.communicate()
    output = output[0].split("\n")

    energy = 0
    
    all_frames = []
    for line in output:
        s = line.split()
        if 'Total Potential Energy' in line:
            energy = float(s[-2].replace('D','e'))
            all_frames.append(energy)
#             print(energy)
    
    return all_frames,len(all_frames)

def run_analyze_arc2(xyzfn):

    cmd1 = f"{tkpath}testgrad {xyzfn} y n n"
    out_log = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE,encoding='utf8')
    output = out_log.communicate()
    data = output[0].split("\n")

    begin_line = np.array([a[1:16] for a in data])
    inds = np.where(begin_line == "Total Potential")[0]

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
    
    return energies, forces, big_forces, len(allen)

def proc_dimer(elfn,ind0,ind1):
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

    indorig = np.arange(ind0,ind1,100)
    gsize = ind1-ind0
    for k,j in enumerate(np.arange(0,gsize,100)):
        if (j+100) > total:
            end = total
        else:
            end = j+100
            
        size = end-j
        dim = alldimers[j:end]
        os.system(f"rm -rf dimer{elfn}.err*")
        np.savetxt(f'dimer{elfn}.xyz',dim,fmt="%s",delimiter='\n')
        en, slen = run_analyze_arc(f'{direc}/dimer{elfn}.xyz')

        slentot = slen
        if len(en) >= 1:
            energies+=en
        
        # print(j,end,slen,len(en))
        j1 = indorig[k]
        j2 = j
        
        if slen == size:
            finalize = 1
        else:
            finalize = 0
            dim2 = dim[slen+1:]
            lsize = 100-slen

            j1 += slen
            j2 += slen
            bad_dimers.append(j1)
            bad_dimers2.append(j2)

        while finalize == 0:
            
            if slen != 0 and slen % 100 == 0:
                break
            
            if lsize <= 0:
                break

            # print(j,end,slen,len(en))
            os.system(f"rm -rf dimer{elfn}.err*")
            
            np.savetxt(f'dimer{elfn}.xyz',dim2,fmt="%s",delimiter='\n')
            en, slen = run_analyze_arc(f'{direc}/dimer{elfn}.xyz')

            if slen != 0:
                slentot += slen
                energies+=en
                try:
                    dim2 = dim2[slen+1:]
                    lsize -= slen
                    j1 += slen
                    j2 += slen
                    bad_dimers.append(j1)
                    bad_dimers2.append(j2)
                except:
                    finalize = 1
            else:
                finalize = 1

    energies = np.array(energies)
    
    if energies.shape[0] > 1:
        np.save(f"energies{elfn}.npy",energies.flatten())
    if len(bad_dimers) > 0:
        np.save(f"bad_dimers{elfn}.npy",np.array(bad_dimers))

        good_dimer = np.delete(alldimers,np.array(bad_dimers2),axis=0)
        np.savetxt(f'dimer{elfn}_final.arc',good_dimer,fmt="%s",delimiter='\n')

    return energies.flatten(), np.array(bad_dimers)

def proc_dimer2(elfn,ind0,ind1):
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
    forces = []
    bad_dimers = []
    bad_dimers2 = []

    indorig = np.arange(ind0,ind1,100)
    gsize = ind1-ind0
    for k,j in enumerate(np.arange(0,gsize,100)):
        if (j+100) > total:
            end = total
        else:
            end = j+100
            
        size = end-j
        dim = alldimers[j:end]
        os.system(f"rm -rf dimer{elfn}.err*")
        np.savetxt(f'dimer{elfn}.xyz',dim,fmt="%s",delimiter='\n')
        en, ff, big_forces, slen = run_analyze_arc2(f'{direc}/dimer{elfn}.xyz')

        slentot = slen
        if len(en) >= 1:
            energies+=en
            forces += ff
        
        # print(j,end,slen,len(en))
        j1 = indorig[k]
        j2 = j
        
        if len(big_forces) > 0:
            for a in big_forces:
                bad_dimers.append((a+j1))
                bad_dimers2.append((a+j2))

        j1 += slen
        j2 += slen
        bad_dimers.append(j1)
        bad_dimers2.append(j2)

        if slen == size:
            finalize = 1
        else:
            finalize = 0
            dim2 = dim[slen+1:]
            lsize = 100-slen
        while finalize == 0:
            
            if slen != 0 and slen % 100 == 0:
                break
            
            if lsize <= 0:
                break
            
            # print(j,end,slen,len(en))
            os.system("rm -rf dimer{elfn}.err*")
            
            np.savetxt(f'dimer{elfn}.xyz',dim2,fmt="%s",delimiter='\n')
            en, ff, big_forces, slen = run_analyze_arc2(f'{direc}/dimer{elfn}.xyz')

            if slen != 0:
                slentot += slen
                energies+=en
                forces += ff
                try:
                    dim2 = dim2[slen+1:]
                    lsize -= slen

                    if len(big_forces) > 0:
                        for a in big_forces:
                            bad_dimers.append((a+j1))
                            bad_dimers2.append((a+j2))
                
                    j1 += slen
                    j2 += slen
                    bad_dimers.append(j1)
                    bad_dimers2.append(j2)
                except:
                    finalize = 1
            else:
                finalize = 1

    energies = np.array(energies)
    forces = np.array(forces)
    
    if energies.shape[0] > 1:
        np.save(f"energies{elfn}_new.npy",energies.flatten())
        np.save(f"forces{elfn}npy",forces)
    if len(bad_dimers) > 0:
        np.save(f"bad_dimers{elfn}_new.npy",np.array(bad_dimers))

        good_dimer = np.delete(alldimers,np.array(bad_dimers2),axis=0)
        np.savetxt(f'dimer{elfn}_final.arc',good_dimer,fmt="%s",delimiter='\n')

    return energies.flatten(), forces, np.array(bad_dimers)

def main():
    os.chdir(direc)

    elfn = int(sys.argv[1])
    ind0 = int(sys.argv[2])
    ind1 = int(sys.argv[3])

    proc_dimer(elfn,ind0,ind1)

if __name__ == "__main__":
    main()