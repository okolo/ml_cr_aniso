import argparse
import train_healpix

cline_parser = argparse.ArgumentParser(description='Calculate minimal detectable fraction',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=56)
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
add_arg('--seed', type=int, help='sample generator seed', default=train_healpix.test_seed)
add_arg('--exposure', type=str, help='exposure: uniform/TA', default='uniform')

args = cline_parser.parse_args()

if len(args.fractions) > 0:
    assert len(args.fractions) == len(args.sources) and len(args.sources) > 1

gen = train_healpix.SampleGenerator(
    args, deterministic=True, sources=args.sources, suffix=args.suffix, seed=args.seed, mixture=args.fractions
)

model = train_healpix.create_model(gen.Ncells, pretrained=args.model)
frac, alpha = train_healpix.calc_detectable_frac(gen, model, args)
print(frac, alpha)

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


