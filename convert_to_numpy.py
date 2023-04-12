import numpy as np
import os
import sys

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def process_data(frm):
    data = frm.split('\n')
    vel = []
    for line in data[1:-1]:
        line2 = line.split()
        v = [float(a.replace('D','e')) for a in line2[-3:]]
        vel.append(v)
    return vel

def convert_velocity(sizefn,velocity_file,path,basenm,Natms,frnumber=1000000,skip=1):

    chunk = os.path.getsize(sizefn)
    with open(velocity_file) as f:            
        aa = read_in_chunks(f,chunk)
        velocities = np.zeros((frnumber,Natms,3),dtype=float)

        if skip != 1:
            c = 0
            for k,frm in enumerate(aa):
                if k % skip != 0:
                    continue
                vel = process_data(frm)
                velocities[c] += vel
                c += 1
                if k % int(skip*100) == 0:
                    print(f"Finished reading {k:6d}...")
                    sys.stdout.flush()

                if k >= frnumber:
                    break
        else:
            for k,frm in enumerate(aa):
                vel = process_data(frm)
                velocities[k] += vel
                
                if k % 100 == 0:
                    print(f"Finished reading {k:6d}...")
                    sys.stdout.flush()

                if k >= frnumber:
                    break

        velocities = np.array(velocities)
        np.save(f"{path}/{basenm}-vel.npy",velocities) ## in Ang/ps

    return velocities

def get_virial(analysis,path,basenm):
    virial_tensor = []

    print("Start reading virial file...\n")
    sys.stdout.flush()

    f = open(analysis,'r')
    pe_data = f.readlines()
    f.close()
    
    sys.stdout.flush()

    begin_lines = [dt[0:4] for dt in pe_data]
    begin_lines = np.array(begin_lines)
    pe_data = np.array(pe_data)
    pe_ind = np.where(begin_lines==' Int')[0]
    
    sys.stdout.flush()
    for k,ind in enumerate(pe_ind):
        
        tt = []
        tv0 = pe_data[ind].strip('\n').split()[-3:]
        tt.append([float(a) for a in tv0])
        
        tv0 = pe_data[ind+1].strip('\n').split()[-3:]
        tv1 = pe_data[ind+2].strip('\n').split()[-3:]
        if len(tv0) == 3 and len(tv1) == 3:
            tt.append([float(a) for a in tv0])
            tt.append([float(a) for a in tv1])
        else:
            for n in range(ind,ind+100,1):
                if begin_lines[n] == ' Pre':
                    break
            
            n = n-3
            tv0 = pe_data[n].strip('\n').split()[-3:]
            tv1 = pe_data[n+1].strip('\n').split()[-3:]
            if len(tv0) == 3 and len(tv1) == 3:
                tt.append([float(a) for a in tv0])
                tt.append([float(a) for a in tv1])
        virial_tensor.append(tt)
    
    sys.stdout.flush()
    del begin_lines, pe_data, pe_ind
            
    virial_tensor = np.array(virial_tensor)
    np.save(f"{path}/{basenm}-virial.npy",virial_tensor)

    return virial_tensor


