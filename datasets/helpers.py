import numpy as np
from pull_shot import ShotPuller

def get_peaking_factors(ne,te):
    te *= 1000*11600 # Convert back to original units
    ne_pf = ne[:,0]/ne.mean(axis=1)
    te_pf = te[:,0]/te.mean(axis=1)

    return ne_pf, te_pf