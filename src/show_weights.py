import matplotlib
import numpy as np
import argparse

beta=0.05

out_file = 'hist.pdf'

cline_parser = argparse.ArgumentParser(description='Spectrum generator')


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('model', type=str, help='model h5 file')
add_arg('--layers', type=int, nargs='*', help='layer numbers counting from 1', metavar='N', default=[1])
add_arg('--show_fig', action='store_true', help="Show learning curves")

args = cline_parser.parse_args()

if not args.show_fig:
    matplotlib.use('Agg')  # enable figure generation without running X server session

import matplotlib.pyplot as plt

import keras

model = keras.models.load_model(args.model)

for i, l in enumerate(args.layers):
    w, b = model.get_layer(index=l).get_weights()
    np.savetxt(args.model + ".L{}.txt".format(l), w)
    plt.subplot(len(args.layers), 1, 1 + i)
    plt.plot(w.flatten(), 'r')
    plt.legend(['Layer ' + str(l)], loc='upper right')

plt.savefig(args.model + ".layers.png")

if args.show_fig:
    plt.show()

