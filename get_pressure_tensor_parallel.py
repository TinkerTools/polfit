import numpy as np
import os
import sys
import math
import multiprocessing, subprocess
from joblib import Parallel, delayed
import time

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
        
        print("check 1")
        sys.stdout.flush()
   
        begin_lines = [dt[0:4] for dt in pe_data]
        begin_lines = np.array(begin_lines)
        pe_data = np.array(pe_data)
        pe_ind = np.where(begin_lines==' Int')[0]
        
        print("check 2")
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
        
        print("check 3")
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

    def calc_pres_tensor(k):
        #data = np.copy(vel_data[k:k+N+1])
        first = 1+k*(N+1)
        last = first+N
        cmd = "sed -n '%d,%dp;%dq' %s" % (first,last,last+1,velocity_file)
        process = subprocess.Popen(cmd, 
            stdout=subprocess.PIPE,shell=True,encoding='utf8')

        data = process.stdout.readlines()
        stdout = process.communicate()[0]

        velocity = []
        for line in data[1:]:
            line2 = line.strip('\n').split()
            v = [float(a.replace('D','e')) for a in line2[-3:]]
            velocity.append(v)

        V = np.array(velocity)
        del velocity
        pres = np.zeros((3,3))
                    
        VR = 4184.0*virial_tensor[k]
        
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
        pressure_tensor[k] = np.array(pres)

        #pressure_tensor = np.array(pressure_tensor) # (J/m3)

        if k % 100 == 0:
            print("Finished %d..." % k)
            sys.stdout.flush()
        del V, pres, mvv, mvv1, data

    Parallel(n_jobs=num_cores)(delayed(calc_pres_tensor)(k) for k in range(virial_tensor.shape[0]))
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
