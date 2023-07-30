import os,sys,time
import subprocess
import signal

#######################
HOME = os.path.expanduser('~')
tinkerpath = f'{HOME}/tinker/bin'

def killjobs(progs,elfn=0):
    username = os.getlogin()
    pslinux = f'/bin/ps aux | grep {username}'
    if elfn != 0:
        pslinux = f' ssh elf{elfn} "/bin/ps aux | grep {username}"'
    proc = subprocess.Popen(pslinux,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = proc.communicate()
    out2 = out.decode(encoding='utf-8')
    output = out2.split('\n')
    jobs = []
    for line in output:
        try:
            res = line.split()
            user = res[0]
            pid = int(res[1])
            comm = " ".join(res[10:])
            jobs.append([pid,comm])
            
        except:
            None
    
    for js in jobs:
        pid,cmd = js
        for pr in progs:
            if pr in cmd:
                if elfn != 0:
                    os.system(f'ssh elf{elfn} "kill -9 {pid}"')
                else:
                    os.kill(pid,signal.SIGKILL)


def gpu_speed(natms,tkpath=tinkerpath):
    tinker9 = f"{tkpath}/tinker9"
    cmd = f"{tinker9} info"
    out_log = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf8')
    output = out_log.communicate()
    all_out = output[0].split('\n')

    cb = 0
    for line in all_out:
        s = line.split()
        if "Maximum compute capability" in line:
            cb = float(s[-1])
            break

    if cb == 0:
        cb = 1

    speed = (0.05+(1000/natms))*50
    if cb > 5.2:
        speed *= (cb-5.2)
    else:
        speed *= (cb/5.2)
    
    gpuspeed = speed

    return gpuspeed

def get_last_frame(fname):    
    if 'gas2.log' in fname:
        cmd = f"""grep "Analysis for" {fname} -a | wc -l"""
    else:
        cmd = f"""grep "Current Time" {fname} -a | wc -l"""
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True,encoding='utf8')
    output = process.stdout.readline()
    stdout = process.communicate()[0]

    try:
        n_lines = int(output.split()[0])
    except:
        n_lines = 0
    
    return n_lines

def count_atoms(fn):
    f = open(fn)
    dt = f.readlines()
    f.close()

    test = dt[2].split()

    if isinstance(test[1], str):
        n = 1
    else:
        n = 2

    c = 0
    for l in dt[n:]:
        if len(l.split()) >= 6:
            c+=1

    return c    

def run_simulation(n,simlen,cmd_liq):
    path = os.getcwd()
    liqproc = subprocess.Popen(cmd_liq, shell=True, universal_newlines='expand_cr')
    filename = os.path.abspath(f"{path}/liquid.log")
    arcfile = os.path.abspath(f"{path}/liquid-{n}.dcd")

    init_time = time.time()

    natms = count_atoms(f"{path}/liquid-{n}.xyz")
    gpuspeed = gpu_speed(natms)
    timsec = simlen * (86.4/gpuspeed)
    
    if simlen < 100:
        sleeper = 30
    else:
        sleeper = 60
    
    timeout = 3*timsec

    running = True
    sucess = False
    trajfn = False
    time.sleep(int(sleeper/3))
    if os.path.isfile(f"{path}/liquid-{n}.err"):
        running = False
    else:
        trajfn = os.path.isfile(arcfile)
        while not trajfn:
            now = time.time()
            if os.path.isfile(f"{path}/liquid-{n}.err"):
                running = False
                break
            trajfn = os.path.isfile(arcfile)
            if now - init_time > sleeper*4:
                running = False
                break
    while running:
        new_last_frame = get_last_frame(filename)
        modtime = os.path.getmtime(arcfile)

        now = time.time()
        run_t = now - init_time
        timer = now - modtime
        if new_last_frame == simlen:
            sucess = True
            break
        if timer > sleeper*5:
            break
        if run_t > timeout:
            break  
        if os.path.isfile(f"./liquid-{n}.err"):
            running = False
            sucess = False
        if new_last_frame == simlen:
            sucess = True
            break
        time.sleep(int(sleeper/2))
    
    if sucess:
        liqproc.communicate()

    killjobs([f'dynamic liquid-{n}'],0)

def main():
    n = int(sys.argv[1])

    simlen = int(sys.argv[2])
    cmd_liq = " ".join(sys.argv[3:])
    
    run_simulation(n,simlen,cmd_liq)

if __name__ == "__main__":
    main()