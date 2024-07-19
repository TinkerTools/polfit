import numpy as np
import importlib

rdkit_loader = importlib.find_loader('rdkit')
found = rdkit_loader is not None

if found:
    from . import gas, liquid, process, prmedit, auxtinker, auxfitting, run_sim, xyz2mol
else:
    from . import gas, liquid, process, prmedit, auxtinker, auxfitting, run_sim

