import numpy as np
import os
import sys
import gc
import subprocess
import time
import analyzetool

sys.path.append("/user/roseane/HIPPO/analyzetool_pkg")
from convert_to_numpy import convert_velocity
from convert_to_numpy import get_virial
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

    
    try:
        skip = int(sys.argv[3])

    except:
        skip = 1

    try:
        tensor_only = str(sys.argv[4])

    except:
        tensor_only = None

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

    compute_all = False
    if tensor_only == None or tensor_only == "None":
        compute_all = True

    if compute_all:
        if os.path.isfile(analysis):
            virial_tensor = get_virial(analysis,path,basenm)
        elif os.path.isfile(f"{path}/{analysis}"):
            virial_tensor = get_virial(f"{path}/{analysis}",path,basenm)
        elif os.path.isfile(f"{path}/{basenm}-virial.npy"):
            virial_tensor = np.load(f"{path}/{basenm}-virial.npy")
        else:
            return

        if skip != 1:
            virial_tensor = virial_tensor[::skip]
        nfrms = virial_tensor.shape[0]
        
        print("Finished processing virial file...\n")
        sys.stdout.flush()
        
        velocity_file = f'{path}/{basenm}.vel'
        if os.path.isfile(velocity_file):
            print("Start reading velocity file...\n")
            sys.stdout.flush()

            sizefn = f"{path}/vel1.txt"
            process = subprocess.Popen(f"tail -{N+1} {velocity_file} > {sizefn} 2>&1", 
                                        shell=True,encoding='utf8')
            
            time.sleep(3)
            velocities = convert_velocity(sizefn,velocity_file,path,basenm,N,nfrms,skip)
        elif os.path.isfile(f"{path}/{basenm}-vel.npy"):
            velocities = np.load(f"{path}/{basenm}-vel.npy")
        else:
            return

    else:
        print("Loading data...\n")
        if os.path.isfile(f"{path}/{basenm}-vel.npy"):
            velocities = np.load(f"{path}/{basenm}-vel.npy")
            if velocities.shape[0] == 250000:
                velocities = velocities[:125000]
                np.save(f"{path}/{basenm}-vel.npy",velocities)
        else:
            return

        if os.path.isfile(f"{path}/{basenm}-virial.npy"):
            virial_tensor = np.load(f"{path}/{basenm}-virial.npy")
        else:
            return
        

    print("Finished processing velocity file...\n")
    sys.stdout.flush()
    nfrms = virial_tensor.shape[0]
    pressure_tensor = np.zeros((nfrms,3,3))
    
    print("Calculating pressure tensor...\n")
    sys.stdout.flush()

    NA=6.02214129*(1e23)
    div = NA*vol
    # num_cores = 1
    # velocity_file = np.load(f"{xyzfile[:-4]}-vel.npy")

    # split_frms = list(range(0,virial_tensor.shape[0],int(virial_tensor.shape[0]/num_cores)))
    # split_frms.append(virial_tensor.shape[0])
    
    def calc_pres_tensor(k):
        pres = np.zeros((3,3))
                    
        VR = 4184.0*virial_tensor[k]
        V = velocities[k]

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

    for k in range(nfrms):
        calc_pres_tensor(k)

    del velocities
    gc.collect()

    np.save(f"{path}/{basenm}-pressure.npy",pressure_tensor)

    # del velocities
    # gc.collect()

if __name__ == "__main__":
    main()
