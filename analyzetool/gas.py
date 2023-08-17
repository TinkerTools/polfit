import numpy as np
from scipy.stats import maxwell
import os

def reject_outliers(data, m = 5.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def convert_float(string):
    a = 0.0
    try:
        a = float(string)
        err = False
    except:
        err = True
        a = 0.0
    return a, err

class GasLog(object):
    def __init__(self, gaspath):
        self.path = gaspath
        self.avgPE = 0
        self.avgKE = 0
        self.stdPE =  0

        self.KE = []
        self.PE = []

        edyn = []
        kdyn = []
        eanl = []
        error = []
        if os.path.isfile(self.path):
            f = open(self.path,'r')
            dt = f.readlines()
            f.close()

            for line in dt:
                s = line.split()
                if len(s) < 2:
                    continue
                if 'Current Potential' in line or 'Potential Energy' in line:
                    try:
                        val, err = convert_float(s[2])
                        edyn.append(val)
                        error.append(err)
                    except:
                        None
                if 'Current Kinetic' in line or 'Kinetic Energy' in line:
                    try:
                        val, err = convert_float(s[2])
                        kdyn.append(val)
                    except:
                        None
                if 'Total Potential Energy : ' in line:
                    try:
                        val, err = convert_float(s[4])
                        eanl.append(val)
                    except:
                        None
        if len(eanl) != 0:
            self.PE = np.array(eanl)
            halfp = int(self.PE.shape[0]/2)

            if halfp > 500:
                halfp = 500

            self.PE = self.PE[halfp:]
            self.avgPE = self.PE.mean()
            self.stdPE = self.PE.std()

            error = [False]

        elif len(edyn) != 0:
            self.PE = np.array(edyn)
            halfp = int(self.PE.shape[0]/2)

            if halfp > 500:
                halfp = 500

            self.H = self.PE[halfp:]+self.KE[halfp:]
            self.avgH = self.H.mean()

            self.PE = self.PE[halfp:]
            self.avgPE = self.PE.mean()
            self.stdPE = self.PE.std()

            self.KE = np.array(kdyn)
            self.KE = reject_outliers(self.KE[halfp:])
            self.avgKE = self.KE.mean()

            

            error.append(False)
        else:
            error.append(True)

        if any(error):
            self.error = True
        else:
            self.error = False

            loc1, scale1 = maxwell.fit(self.PE)
        
            off = 0.1

            m2,v2 = maxwell.stats(loc=loc1,scale=(scale1-off))
            self.avgPE = m2
            self.stdPE = np.sqrt(v2)

