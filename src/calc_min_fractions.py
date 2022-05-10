import argparse

import numpy as np

from train import test_seed, f_sampler

cline_parser = argparse.ArgumentParser(description='Calculate minimal detectable fraction',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=28)
add_arg('--EminData', type=float, help='minimal data energy in EeV', default=56)
# add_arg('--source_id', type=str,
#         help='source (CenA, NGC253, M82, M87 or FornaxA) or comma separated list of sources or "all"',
#         default='CenA')
add_arg('sources', type=str, nargs='+', metavar='source', default=[])
add_arg('--fractions', type=float, nargs='+', metavar='frac', help='fractions for mixed source case (in the same order as sources)', default=[])
add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=32)
add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
add_arg('--source_vicinity_radius', type=str, help='source vicinity radius', default='1')
add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)
add_arg('--model', type=str, help='healpix NN', default='')
add_arg('--n_samples', type=int, help='number of samples', default=50000)
add_arg('--alpha', type=float, help='type 1 maximal error', default=0.01)
add_arg('--beta', type=float, help='type 2 maximal error', default=0.05)
add_arg('--suffix', type=str, default='*')
add_arg('--batch_size', type=int, help='size of training batch', default=100)
add_arg('--mf', type=str, help='Magnetic field model (jf or pt)', default='jf')
add_arg('--data_dir', type=str, help='data root directory (should contain jf/sources/ or pt/sources/)',
            default='data')
add_arg('--threshold', type=float,
            help='source fraction threshold for binary classification', default=0.0)
add_arg('--sigmaLnE', type=float, help='deltaE/E energy resolution', default=0.2)
add_arg('--seed', type=int, help='sample generator seed', default=test_seed)
add_arg('--exclude_energy', action='store_true', help='legacy mode without energy as extra observable')
add_arg('--exposure', type=str, help='exposure: uniform/TA', default='uniform')
add_arg('--n_iterations', type=int, help='increase number of iterations for more precise result', default=1)

args = cline_parser.parse_args()
if len(args.fractions) > 0:
    assert len(args.fractions) == len(args.sources) and len(args.sources) > 1

try:
    import train_healpix
    model = train_healpix.create_model(12*args.Nside*args.Nside, pretrained=args.model)
    Generator = train_healpix.SampleGenerator
except:
    import train_gcnn
    model = train_gcnn.create_model(args.Neecr, pretrained=args.model)
    Generator = train_gcnn.SampleGenerator

test_batches = (args.seed == test_seed)
if not test_batches:
    np.random.seed(args.seed)

gen = Generator(
        args, deterministic=test_batches, sources=args.sources, suffix=args.suffix, seed=args.seed, mixture=args.fractions)

from beta import calc_detectable_frac
frac, alpha = calc_detectable_frac(gen, model, args, n_iterations=args.n_iterations)

out_file = args.model + "_cmp.txt"
with open(out_file, "a") as d:
    print("Model to compare with:", file=d)
    print(*args.sources, file=d)
    if len(args.fractions) > 0:
        print('fractions:', *args.fractions, file=d)
    print("Neecr={:3d}".format(args.Neecr), file=d)
    print("Nmixed_samples={:5d}".format(args.n_samples), file=d)
    d.write("frac={:7.2f}\n".format(frac*100))
    d.write("alpha={:6.4f}\n".format(alpha))
    d.write("------------------------------\n")
