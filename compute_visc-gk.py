import numpy as np
import os, sys
import analyzetool
import scipy


viscd = "/work/roseane/HIPPO/small_molecules/benzene/simulations/temperature_dependance/viscosity/single-conf"
def compute_visc(skip=1,pp=None,path=None):
    T = 298.15
    ni=0
    KB_J = 1.38064852e-23 #J/K

    if pp != None:
        simd = f"{viscd}/run-{pp}"
    elif path != None:
        simd = path
    else:
        return

    if os.path.isfile(f"{simd}/liquid.xyz"):
        xyzf = "liquid"
    if os.path.isfile(f"{simd}/visc.xyz"):
        xyzf = "visc"
    
    if os.path.isfile(f"{simd}/{xyzf}-pressure.npy") and os.path.isfile(f"{simd}/{xyzf}-pressure-2.npy"):
        p1 = np.load(f"{simd}/{xyzf}-pressure.npy")
        p2 = np.load(f"{simd}/{xyzf}-pressure-2.npy")
        pressure_tensor = np.concatenate((p1,p2))
        total = 3000-0.01
    # else:
    #     return
    elif os.path.isfile(f"{simd}/{xyzf}-pressure-2.npy"):
        pressure_tensor = np.load(f"{simd}/{xyzf}-pressure-2.npy")
        total = 2000-0.01
    else:
        return
    
    arcfile = analyzetool.process.ARC(f"{simd}/{xyzf}.xyz")
    arcfile.read_xyz()
    arcfile.compute_volume()
        
    vol = (1e-30)*arcfile.volume[0]
    
#     pressure_tensor = np.load(f"{simd}/liquid-pressure.npy")

    N = pressure_tensor.shape[0]
    tstep = 0.01
    time_step = tstep*(1e-12)

    # Calculate the off-diagonal elements of the pressure tensor
    P_shear = np.zeros((6,N), dtype=float)
    P_shear[0] = pressure_tensor[:,0,1]
    P_shear[1] = pressure_tensor[:,0,2]
    P_shear[2] = pressure_tensor[:,1,2]
    P_shear[3] = (pressure_tensor[:,0,0] - pressure_tensor[:,1,1]) / 2  # xx-yy
    P_shear[4] = (pressure_tensor[:,1,1] - pressure_tensor[:,2,2]) / 2  # yy-zz
    P_shear[5] = (pressure_tensor[:,0,0] - pressure_tensor[:,2,2]) / 2  # xx-zz

    # At increasing time lengths, calculate the viscosity based on that part of the simulatino
    size = N-int(500/tstep)
    ACF = np.zeros((6,size), dtype=float)
    for t in range(size):
        ACF[:,t] = np.mean(P_shear[:,:N-t] * P_shear[:,t:])
    ACF_avg = ACF.mean(axis=0)

    # Finally calculate the overall viscosity
    # Note that here the first step is skipped to avoid divide by zero issues
    kbT = KB_J * T
    ACF_avg *= vol / (kbT)

    visc = scipy.integrate.cumtrapz(y=ACF_avg, dx=time_step)
    
   
    # Print the final viscosity
    if pp != None:
        print(f"{pp:3d} Viscosity is {(1e3)*visc[ni:].mean():7.2f} cP")
    else:
        print(f"Viscosity is {(1e3)*visc[ni:].mean():7.2f} cP")
    np.save(f"{simd}/{xyzf}-visc-gk-2.npy",visc)


def main():
    try:
        pp = int(sys.argv[1])
        path = None
    except:
        pp = str(sys.argv[1])
        path = os.path.abspath(pp)
        if os.path.isdir(path):
            pp = None
        else:
            return

    try:
        skip = int(sys.argv[2])

    except:
        skip = 1

    compute_visc(skip,pp,path)

if __name__ == "__main__":
    main()