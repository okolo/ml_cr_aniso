import train_spec
import argparse

def files_from_mask_list(mask_list):
    import glob
    for mask in mask_list:
        yield from glob.glob(mask)

cline_parser = argparse.ArgumentParser(description='Calculate neural network model based statistics')
def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('--model', type=str, help='model h5 file', required=True)
add_arg('data', type=str, nargs='*', help='npz file list or mask')
add_arg('--features', type=str, nargs='*', metavar='F', help='features to use (currently available spectrum,alm,raw)', default=train_spec.feature_names)
add_arg('--skip_normalization', action='store_true', help="Do not normalize data")

args = cline_parser.parse_args()

import numpy as np
import keras
from os import path

model_file = args.model
model = keras.models.load_model(model_file)
n_features = int(model.input_shape[1])
train_spec.feature_names = args.features

for npz in files_from_mask_list(args.data):
    out_file = path.basename(model_file).replace('.h5','') + "__" + path.basename(npz)
    frac, features = train_spec.load_features(npz)
    if features.shape[1] != n_features:
        print('skipping', npz, ': incompatible model (number of input features)')
        continue
    if not args.skip_normalization:
        features = train_spec.normalize_features(features)
    frac_pred = model.predict(features)
    print('saving data to', out_file)
    np.savez(out_file, frac=frac, xi=frac_pred.flatten())

