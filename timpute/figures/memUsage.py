# import psutil
import pickle
import os
import numpy as np
import resource
import argparse

from figures import METHODS, METHODNAMES, SAVENAMES
from ..decomposition import Decomposition
from ..generateTensor import generateTensor

# poetry run python -m timpute.figures.dataUsage

def createDictionary(filename='nonmissing'):
    memory = dict()
    for i in SAVENAMES:
        memory[i] = dict()
        for j in METHODNAMES:
            memory[i][j] = []

    with open('./timpute/figures/cache/dataUsage/'+filename+'.pickle', 'wb') as handle:
        pickle.dump(memory, handle)

    return True


def testMemory(dataname, method, methodname, max_comp, filename,
               dropType='entry', dropPerc=0, seed=1):
    # generate tensor
    np.random.seed(seed)
    orig = generateTensor(type=dataname)

    # set nonmissing case
    decomposition = Decomposition(orig, max_comp)
    
    # include **kwargs include: repeat=reps, drop=drop_perc, init=init_type
    decomposition.profile_imputation(type=dropType, method=method, drop=float(dropPerc))
    # decomposition.save(f"./timpute/figures/cache/dataUsage/{dataname}-{method.__name__}.decomposition")

    if os.path.isfile('./timpute/figures/cache/dataUsage/'+filename+'.pickle') is False:
        createDictionary(filename)
    
    with open('./timpute/figures/cache/dataUsage/'+filename+'.pickle', 'rb') as handle:
        memory = pickle.load(handle)
    memory[dataname][methodname].append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    with open('./timpute/figures/cache/dataUsage/'+filename+'.pickle', 'wb') as handle:
        pickle.dump(memory, handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile Imputation')
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--dropType', required=False)
    parser.add_argument('--dropPerc', required=False)
    parser.add_argument('--seed', required=False)
    args = parser.parse_args()
    
    if args.dataname == 'zohar':
        i = 0
    elif args.dataname == 'alter':
        i = 1
    elif args.dataname == 'hms':
        i = 2
    elif args.dataname == 'coh_response':
        i = 3

    if args.method == 'DO':
        j = 0
    elif args.method == 'ALS':
        j = 1
    elif args.method == 'CLS':
        j = 2

    max_comps = [10,10,10,20]
    # resource.setrlimit(resource.RLIMIT_AS, (int(1e9), int(1e9)))
    if args.dropType and args.dropPerc and args.seed:
        testMemory(SAVENAMES[i], METHODS[j], METHODNAMES[j], max_comps[i], args.filename,
                   args.dropType, float(args.dropPerc), int(args.seed))
    else:
        testMemory(SAVENAMES[i], METHODS[j], METHODNAMES[j], max_comps[i], args.filename)
        
    