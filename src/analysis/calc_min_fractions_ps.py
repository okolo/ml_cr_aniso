#
# The scripts calculates quantities defined in Tables 1 and 2 of the JCAP paper
#
import glob
import numpy as np
import argparse
from sys import stderr

out_file = 'hist.pdf'

cline_parser = argparse.ArgumentParser(description='Calculate minimal detectable from-source event fraction')


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('--iso', type=str, nargs='*', metavar='npz_file', help='iso npz file(s) or mask', required=True)
add_arg('--mixed', type=str, nargs='*', metavar='npz_file', help='mixed npz file(s) or mask', required=True)
add_arg('--alpha', type=float, help='type 1 maximal error', default=0.01)
add_arg('--beta', type=float, help='type 2 maximal error', default=0.05)

args = cline_parser.parse_args()


def load_npz(mask_list):
    for mask in mask_list:
        for f in glob.glob(mask):
            try:
                yield np.load(f)
            except Exception:
                print('skipping', f)

def load_data(mask_list):
    npzs = list(load_npz(mask_list))
    assert len(npzs) > 0, 'no valid data found in files matching mask list'

    xi = [npz['xi'] for npz in npzs]
    fractions = [npz['frac'] for npz in npzs]

    if len(npzs) == 1:
        xi = xi[0]
        fractions = fractions[0]
    else:
        xi = np.hstack(xi)
        fractions = np.hstack(fractions)
    #idx = np.argsort(fractions)
    #return fractions[idx], xi[idx]
    return fractions, xi


iso_fractions, iso_xi = load_data(args.iso)
assert np.max(iso_fractions) == 0.
fractions, xi = load_data(args.mixed)


if np.mean(iso_xi) > np.mean(xi):  # below we assume <iso_xi>  <=  <xi>
    xi *= -1.
    iso_xi *= -1.

alpha_thr = np.quantile(iso_xi, 1.-args.alpha)

# sort by xi
idx = np.argsort(xi)
fractions = fractions[idx]
xi = xi[idx]


thr_idx = np.where(xi >= alpha_thr)[0][0]

fracs = np.array(sorted(list(set(fractions))))

beta = np.zeros_like(fracs)

for i_f, f in enumerate(fracs):
    idx = np.where(fractions == fracs[i_f])[0]
    idx_left = np.where(idx < thr_idx)[0]
    beta[i_f] = len(idx_left)/len(idx)

th_eta = 1.

i = np.where(beta < args.beta)[0]
if len(i) > 0:
    th_eta = fracs[i[0]]

table = np.hstack((fracs.reshape((-1, 1)), beta.reshape((-1, 1))))
output_file_name = args.mixed[0].replace('.npz','').replace('*','XXX') + '.beta'
np.savetxt(output_file_name, table, fmt='%g', header='#frac\tbeta')

print(th_eta)

exit(0)
