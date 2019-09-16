import matplotlib
import glob
import numpy as np
import argparse

beta=0.05

out_file = 'hist.pdf'

cline_parser = argparse.ArgumentParser(description='Spectrum generator')


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('masks', type=str, nargs=2, help='npz file mask(s) or comma separated lists', default=['*__*.npz','*__*.npz'])
add_arg('--f1', type=float, nargs=2, help='range of fractions for hist 1 [Fmin, Fmax)', metavar=('Fmin', 'Fmax'), default=[0., 1.])
add_arg('--f2', type=float, nargs=2, help='range of fractions for hist 2 [Fmin, Fmax)', metavar=('Fmin', 'Fmax'), default=[0., 1.])
add_arg('--show_fig', action='store_true', help="Show learning curves")

args = cline_parser.parse_args()

if not args.show_fig:
    matplotlib.use('Agg')  # enable figure generation without running X server session

import matplotlib.pyplot as plt


def load_npz(mask_list):
    for mask in mask_list.split(','):
        for f in glob.glob(mask):
            try:
                yield np.load(f)
            except Exception:
                print('skipping', f)

def load_data(mask_list, fr_range):
    npzs = list(load_npz(mask_list))
    assert len(npzs) > 0, 'no valid data found in files matching mask list' + mask_list

    xi = [npz['xi'] for npz in npzs]
    fractions = [npz['frac'] for npz in npzs]

    if len(npzs) == 1:
        xi = xi[0]
        fractions = fractions[0]
    else:
        xi = np.hstack(xi)
        fractions = np.hstack(fractions)

    if len(fr_range)==2:
        idx = np.where((fractions >= fr_range[0]) & (fractions < fr_range[1]))[0]
        xi = xi[idx]

    return xi


data = [load_data(mask, r) for mask, r in zip(args.masks, (args.f1, args.f2))]

min_xi = min([min(d) for d in data])
max_xi = max([max(d) for d in data])

bins = np.linspace(min_xi, max_xi, 100)

for i, d in enumerate(data):
    plt.hist(d, bins, alpha=0.5, label=str(i))

plt.legend(loc='upper right')

plt.savefig(out_file)
print('histogram was saved in', out_file)

if len(data) == 2:
    av = [np.mean(d) for d in data]
    if av[0] > av[1]:
        data = [-d for d in data]
    thr = np.quantile(data[1], beta)
    alpha = len(np.where(data[0] > thr)[0])/len(data[0])
    print('alpha =', alpha)

if args.show_fig:
    plt.show()
