import sys
import glob

import numpy as np
from os import path

if len(sys.argv) < 2:
    print('usage: python', sys.argv[0], 'file1.npz [file2.npz ..]')
    sys.exit(1)

#input = 'aps_CenA_D3.5_Emin56_Neecr500_Nsample1024_R1_Nside512_*.npz'
iso_input = 'iso_Neecr500_Nsample1024_Nside512_*.npz'

def load_npz(mask):
    for f in glob.glob(mask):
        try:
            yield np.load(f)
        except Exception:
            print('skipping', f)

def load_data(mask):
    npzs = list(load_npz(mask))
    assert len(npzs) > 0, 'no valid data found in files matching mask ' + mask

    spectrum = [npz['spectrum'] for npz in npzs]
    fractions = [npz['fractions'] for npz in npzs]

    if len(npzs) == 1:
        spectrum = spectrum[0]
        fractions = fractions[0]
    else:
        spectrum = np.vstack(spectrum)
        fractions = np.hstack(fractions)

    return spectrum, fractions

iso_spec, _ = load_data(iso_input)
mean_iso = np.mean(iso_spec, axis=0, keepdims=True)
sigma_iso = np.std(iso_spec, axis=0, keepdims=True)

# define function of spectrum to use, e.g. D
def f(spec):
    D = (spec - mean_iso)/sigma_iso
    D = np.sum(D, axis=1)/spec.shape[1]
    return D

for npz in sys.argv[1:]: # glob.glob(input):
    out_file = 'f__' + path.basename(npz)
    npz = np.load(npz)
    data = npz['spectrum']
    if iso_spec.shape[1] != data.shape[1]:
        print('skipping data with incompatible shape', npz)
        continue

    xi = f(data)
    frac = npz['fractions']
    print('saving data to', out_file)
    np.savez(out_file, frac=frac, xi=xi)

