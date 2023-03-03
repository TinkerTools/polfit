import time, sys
import numpy as np
import os
from numpy import linalg as LA

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

def compute_torque(forces,pmi,coords,xcm,pair_index):
    
    torques = []
    for q in range(forces.shape[0]):
        i = pair_index[q][0]
        j = pair_index[q][1]

        x = np.copy(coords[i])
        xicm = np.copy(xcm[i])

        fatm = forces[q]
        
        trq = np.zeros((3))

        fcmi = np.array(fatm).sum(axis=0)

        rcmx = np.zeros((3,3))
        rcmx[:,0] = x[:,0] - xicm[0]
        rcmx[:,1] = x[:,1] - xicm[1]
        rcmx[:,2] = x[:,2] - xicm[2]

        trqx = rcmx[:,1]*fatm[:,2]-rcmx[:,2]*fatm[:,1]
        trqy = rcmx[:,2]*fatm[:,0]-rcmx[:,0]*fatm[:,2]
        trqz = rcmx[:,0]*fatm[:,1]-rcmx[:,1]*fatm[:,0]

        trq[0]+=trqx.sum()
        trq[1]+=trqy.sum()
        trq[2]+=trqz.sum()
        
        tq = np.power(trq,2)
        
        tt = (tq[0]/pmi[i][0])+(tq[1]/pmi[i][1])+(tq[2]/pmi[i][2])
        torques.append(tt)
        
    return np.array(torques)

def main():
    base_dir = "/work/roseane/HIPPO/virial2_data/trial18-current/sim_298"
    direc = "/work/roseane/HIPPO/virial2_data/trial18-current/sim_298"
    pair_index = np.load(f"{direc}/pair_index.npy")
    pmi = np.load(f"{direc}/pmi.npy")
    xcm = np.load(f"{direc}/xcm.npy")
    coords = np.load(f"{direc}/coords.npy")

    # pair_index = pair_index[:13316400]

    rdist = float(sys.argv[1])

    base_dir = '/work/roseane/HIPPO/virial2_data/trial18-current/sim_298'
    rdir = f"{base_dir}/{rdist:.2f}_calc"

    os.chdir(rdir)
    
    print(f"Starting r = {rdist}")
    sys.stdout.flush()
    
    eng = np.load(f"{rdir}/inter_energies.npy")
    forces = np.load(f"{rdir}/forces.npy")

    mask1 = forces[:,:,-1] < 1000
    mask2 = mask1.sum(axis=1)
    mask3 = eng > -10
    mask4 = mask3 + mask2
    mask = mask4 == 7

    del mask1, mask2, mask3, mask4
    
    forces = forces[:,:3,:-1][mask]
    eng = eng[mask]
    
    print(forces.shape[0], pair_index.shape[0])
    pind = pair_index[mask]

    tq = compute_torque(forces,pmi,coords,xcm,pind)   
    np.save(f"{rdir}/tq_pmi.npy",tq)

if __name__ == "__main__":
    main()