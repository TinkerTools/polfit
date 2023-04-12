import numpy as np
import os
import sys
import subprocess
import analyzetool
from time import sleep

sys.path.append("/user/roseane/HIPPO/analyzetool_pkg")
from convert_to_numpy import get_virial
from convert_to_numpy import read_in_chunks, process_data
# from compute_visc import compute_visc

def main():
    try:
        xyzfile = str(sys.argv[1])

        if xyzfile[-4:] != ".xyz":
            xyzfile += ".xyz"
    except:
        xyzfile = "liquid.xyz"

    try:
        analysis = str(sys.argv[2])

    except:
        analysis = "analysis.log"

    
    if os.path.isfile(xyzfile):
        print("Starting...\n")
        sys.stdout.flush()

        arcfile = analyzetool.process.ARC(xyzfile)
        arcfile.read_xyz(xyzfile)
        arcfile.compute_volume()
        m_atms = arcfile.atom_map

        vol = (1e-30)*arcfile.volume[0]
        basenm = xyzfile.split('/')[-1][:-4]
        path = "/".join(os.path.abspath(xyzfile).split('/')[:-1])
        N = int(arcfile.n_atoms)
    else:
        print("The given xyz file does not exist\n")
        sys.stdout.flush()
        return

    
    if os.path.isfile(analysis):
        virial_tensor = get_virial(analysis,path,basenm)
    elif os.path.isfile(f"{path}/{analysis}"):
        virial_tensor = get_virial(f"{path}/{analysis}",path,basenm)
    else:
        return

    nfrms = virial_tensor.shape[0]

    if nfrms == 10020:
        virial_tensor=virial_tensor[20:]
        nfrms = virial_tensor.shape[0]

    print("Finished processing virial file...\n")
    sys.stdout.flush()
    
    velocity_file = f'{path}/{basenm}.vel'
    if not os.path.isfile(velocity_file):
        return
    print("Start reading velocity file...\n")
    sys.stdout.flush()

    sizefn = f"{path}/vel1.txt"
    process = subprocess.Popen(f"tail -{N+1} {velocity_file} > {sizefn} 2>&1", 
                                shell=True,encoding='utf8')
    
    sleep(3)

    chunk = os.path.getsize(sizefn)
    
    NA=6.02214129*(1e23)
    div = NA*vol
    pressure_tensor = np.zeros((nfrms,3,3))
    
    print("Calculating pressure tensor...\n")
    sys.stdout.flush()


    def calc_pres_tensor(k,V):
        pres = np.zeros((3,3))
                    
        VR = 4184.0*virial_tensor[k]
        # V = velocities[k]

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
        pres[1,2] += (10*mvv1 - VR[1,2])
        pres[2,1] += (10*mvv1 - VR[1,2])

        pres /= div
        pressure_tensor[k] = np.array(pres)

        if k % 100 == 0:
            print(f"Finished {k:6d}...")
            sys.stdout.flush()

    with open(velocity_file) as f:            
        aa = read_in_chunks(f,chunk)

        for k,frm in enumerate(aa):
            vel = process_data(frm)
            
            calc_pres_tensor(k,np.array(vel))

            if k >= nfrms:
                break

    np.save(f"{path}/{basenm}-pressure-2.npy",pressure_tensor)

if __name__ == "__main__":
    main()
