""" The script calculates the angular power spectrum for Nsample samples
of a mixed flux with a fraction coming from a predefined source.
It is to be used then by plot_mean_spectra.py and plot_D2.py

NB: the EZ_threshold part has not been updated for different methods
of calculating the healpix map and for including N nuclei in a cell.

"""

from __future__ import print_function, division

import sys
import numpy as np
import healpy as hp
#from backports import lzma
import lzma
from os import path, makedirs
import argparse
#______________________________________________________________________
normalize_aps = False

# fraction of "from-source" EECRs (0,1];
Fsrc = None

cline_parser = argparse.ArgumentParser(description='Spectrum generator')


def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


add_arg('--output_dir', type=str, help='data file prefix (or comma separated list of prefixes)', default='.')
add_arg('--f_src', type=float, help='fraction of "from-source" EECRs [0,1] or -1 for random', default=-1)
add_arg('--Neecr', type=int, help='Total number of EECRs in each sample', default=500)
add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=56)
add_arg('--Nmixed_samples', type=int, help='Number of mixed samples (i.e., the sample size)', default=1000)
add_arg('--source_id', type=str, help='source (CenA, NGC253, M82, M87 or FornaxA)', default='CenA')
add_arg('--data_dir', type=str, help='data root directory (should contain jf/sources/ or pt/sources/)', default='data')
add_arg('--mf', type=str, help='Magnetic field model (jf or pt)', default='jf')
add_arg('--start_idx', type=int, help='file idx to start from', default=0)
add_arg('--alm', action='store_true', help="generate a_lm")
add_arg('--raw', action='store_true', help="generate raw healpix data")
add_arg('--l_max', type=int, help='file idx to start from', default=32)
add_arg('--Nside', type=int, help='healpix grid Nside parameter', default=512)
add_arg('--Nini', type=int, help='Size of the initial sample of from-source events', default=10000)
add_arg('--raw_only', action='store_true', help="generate only raw healpix data")
add_arg('--max_data_size', type=float, help='maximal raw data size in Gb', default=1.)
add_arg('--log_sample', action='store_true', help="sample f_src uniformly in log scale")
add_arg('--f_src_max', type=float, help='maximal fraction of "from-source" EECRs [0,1]', default=1)
add_arg('--f_src_min', type=float, help='minimal fraction of "from-source" EECRs [0,1]', default=0)


args = cline_parser.parse_args()

if args.raw_only:
    args.raw = True

if not path.isdir(args.output_dir):
    makedirs(args.output_dir)

Neecr = args.Neecr

if 0 <= args.f_src <= 1.:
    Fsrc = args.f_src
else:
    assert 0 <= args.f_src_min < 1
    assert 0 < args.f_src_max <= 1
    assert args.f_src_min < args.f_src_max

    N_src_min = np.round(args.f_src_min * Neecr)
    N_src_max = np.round(args.f_src_max * Neecr)
    if N_src_min == N_src_max:
        Fsrc == N_src_min/Neecr
    elif args.log_sample:
        if args.f_src_min == 0:
            # make sure roughly equal amount of isotropic and mixture samples are generated
            logNmin = np.log( 1. / N_src_max)
        else:
            logNmin = np.log(args.f_src_min * Neecr)

        logNmax = np.log(args.f_src_max * Neecr)


Nmixed_samples = args.Nmixed_samples

# Emin for which the input sample was generated
Emin = args.Emin   # EeV

source_id = args.source_id

# Calculate the HEALPix map "as is" with 1s in cells with nuclei (or,
# possibly, the number of nuclei) and zeros otherwise OR as the
# relative intensity = (n_i-m_i)/m_i, where n_i is the number of
# nuclei in the ith cell, and m_i being the expected (reference)
# number of nuclei in the cell OR as in the Auger article 1611.06812.
#
# Should be either "asis" (default) or "relins" or "auger"
healpix_map_method = 'asis'

# Less used initial parameters
# healpix grid parameter
Nside = args.Nside

# l_max for the angular power spectrum plot
lmax = args.l_max

# Radius of the vicinity of a source used when making a sample
source_vicinity_radius = 1

random_seed = 2**27-10000

# Size of the initial sample of from-source events. It is used
# in the initial file name and when making a sample of Fsrc*N_EECR
# IT SHOULD NOT BE MODIFIED UNLESS A NEW INPUT FILE IS CREATED
Nini = args.Nini


if source_id=='M82':
    source_lon = 141.4095
    source_lat = 40.5670
    D_src = '3.5'    # Mpc
elif source_id=='CenA':
    source_lon = 309.5159
    source_lat = 19.4173
    D_src = '3.5'    # Mpc
elif source_id=='NGC253':
    source_lon = 97.3638
    source_lat = -87.9645
    D_src = '3.5'    # Mpc
elif source_id=='NGC6946':
    source_lon = 95.71873
    source_lat = 11.6729
    D_src = '6.0'
elif source_id=='M87':
    source_lon = 283.7777
    source_lat = 74.4912
    D_src = '18.5'    # Mpc
elif source_id=='FornaxA':
    source_lon = 240.1627
    source_lat = -56.6898
    D_src = '20.0'    # Mpc
else:
    print('\nUnknown source!')
    sys.exit()

#______________________________________________________________________

print(source_id)
print('Emin  = ' + str(Emin) + ' EeV')
print('Nside = ' + str(Nside))
print('lmax  = ' + str(lmax))
print('Neecr = ' + str(Neecr))
print('Nmixed_samples = ' + str(Nmixed_samples))
print('Healpix map method: ' + healpix_map_method)

suffix = ''
if args.raw:
    suffix += '_raw'
if normalize_aps:
    suffix += '_norm'

print('\n')

# Input file name; the file must be prepared with src_sample.py
# or convert_src_sampe.py.
# It provides data for making a sample of from-source events
infile = ('src_sample_' + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + '_N' + str(Nini)
        + '_R' + str(source_vicinity_radius)
        + '_Nside' + str(Nside)
        + '.txt.xz')

# File for individual spectra
#outfile1 = ('data/mixed_spectra/individual_mixed_spectra_'
if Fsrc is not None:
    if Fsrc == 0.:
        outfile1 = ('iso'
                    + '_Neecr' + str(Neecr)
                    + '_Nsample' + str(Nmixed_samples)
                    + '_Nside' + str(Nside)
                    + suffix)
    else:
        outfile1 = ('aps_'
                    + source_id + '_D' + D_src
                    + '_B' + args.mf
                    + '_Emin' + str(Emin)
                    + '_Neecr' + str(Neecr)
                    + '_Nsample' + str(Nmixed_samples)
                    + '_Fsrc{:02d}'.format(int(round(Fsrc * 100)))
                    + '_R' + str(source_vicinity_radius)
                    + '_Nside' + str(Nside)
                    + suffix)
else:
    if args.log_sample:
        suffix += '_logF_src'
    if args.f_src_min > 0:
        suffix += '_f_min' + str(args.f_src_min)
    if args.f_src_max < 1:
        suffix += '_f_max' + str(args.f_src_max)
    outfile1 = ('aps_'
                + source_id + '_D' + D_src
                + '_B' + args.mf
                + '_Emin' + str(Emin)
                + '_Neecr' + str(Neecr)
                + '_Nsample' + str(Nmixed_samples)
                + '_R' + str(source_vicinity_radius)
                + '_Nside' + str(Nside)
                + suffix)


# We need to know the expected flux in a healpix cell if we are
# calculating the relative intensity map or an Auger-like map
#Ncells = 12*Nside**2
Ncells = hp.nside2npix(Nside)

if args.raw:
    if Ncells*Nmixed_samples/2**30 > args.max_data_size:
        Nmixed_samples = args.max_data_size * 2**30 / Ncells
        print('number of samples reduced to', Nmixed_samples)

assert Nmixed_samples>0

# A template for C_l, alm, healpix_map one line per each isotropic sample
spectrum = None
alm = None
raw = None
if not args.raw_only:
    spectrum = np.zeros((Nmixed_samples,lmax+1))
if args.raw:
    raw = np.zeros((Nmixed_samples, Ncells), dtype=np.uint8)  # save space by using one byte type

fractions = np.zeros((Nmixed_samples,))

for i in range(args.start_idx, args.start_idx + 100000):
    out = args.output_dir + '/' + outfile1 + '_' + str(i) + '.npz'
    if not path.isfile(out):
        with open(out, 'x') as lock_file:  # lock file to avoid overwriting
            outfile1 = out
            np.random.seed(random_seed + i)  # make file content deterministic
        break
#______________________________________________________________________

# The number of from-source and isotropic-background events
# Nsrc = int(np.round(float(Neecr)*Fsrc))
# Niso = Neecr - Nsrc

# 1. Read a file produced by src_sample.py:
#    0        1        2        3       4     5  6     7
# lat_ini, lon_ini, lat_res, lon_res, angsep, Z, E, cell_no
#data = np.loadtxt('data/sources/'+infile)
data = []
if Fsrc != 0.:
    try:
        infile = args.data_dir + '/' + args.mf + '/sources/' + infile
        with lzma.open(infile, 'rt') as f:
            data = np.genfromtxt(f,dtype=float)
            # data = np.loadtxt('data/sources/'+infile)
    except IOError:
        print('\n-------> ' + infile + ' file not found!')
        print('-------> Create one with [convert_]src_sample.py\n')
        sys.exit()

# if healpix_map_method=='relins':
#     # expected number of Neecr per cell
#     expected_flux = float(Neecr)/Ncells
# elif healpix_map_method=='auger':
#     K = 4*np.pi/Neecr   #/Ncells


# 2. Find non-zero lines, i.e., those with Z>0:
if Fsrc != 0.:
    tp = np.arange(0,np.size(data[:,0]))
    nonz = tp[data[:,5]>0]
    nonz_number = len(nonz)

    # These are numbers of cells on the healpix grid occupied by the
    # nuclei in the infile
    healpix_src_cells = data[nonz,7].astype(int)
    data = []

if Fsrc is not None:
    Nsrc = int(np.round(float(Neecr) * Fsrc))
    Niso = Neecr - Nsrc
#______________________________________________________________________
# Here we create Nmixed_samples of the mixed flux and calculate
# their angular power spectra



print('Generating samples and calculating spectra...')
for i in np.arange(0,Nmixed_samples):

    if np.mod(i+1,100)==0:
        print(i+1)

    # A template for the HEALPix map
    if healpix_map_method=='asis':
        # Previously, it was dtype=int. Is this important?
        healpix_map = np.zeros(Ncells, dtype=float)
    else:
        # This is for "relins". What should be for Auger?
        healpix_map = -1*np.ones(Ncells, dtype=float)

    if Fsrc is None:
        if args.log_sample:
            Nsrc = int(np.round(np.exp(logNmin + (logNmax-logNmin)*np.random.rand())))
            assert N_src_min <= Nsrc <= N_src_max
        else:
            Nsrc = np.random.randint(N_src_min, N_src_max+1)

        Niso = Neecr - Nsrc

    if Nsrc > 0:
        # Create a random sample of from-source events:
        if nonz_number>Nsrc:
            src_sample = np.random.choice(nonz_number,Nsrc,replace=False)
        elif nonz_number<Nsrc:
            print('The input sample of from-source events is small:')
            print('We need Nsrc={:4d} but only have {:4d}'.
                    format(Nsrc,nonz_number))
            src_sample = np.random.choice(nonz_number,Nsrc,replace=True)
        else:
            src_sample = nonz

        # Cells of the healpix grid "occupied" by the from-source sample
        src_cells = healpix_src_cells[src_sample]

        # This is a simple way to define the map. It assumes
        # only one EECR gets in a cell. Works OK for Nside=512.
        #healpix_map[src_cells] = 1

        # "cells" is an intermediate variable to simplify the code.
        # It consists of two arrays: [0] is a list of unique cells,
        # [1] is their multiplicity.
        cells = np.unique(src_cells,return_counts=True)
        healpix_map[cells[0]] += cells[1]

    if Niso > 0:
        # A sample of events from the isotropic background
        #np.random.seed(iso_random_seed)
        lon_iso = np.random.uniform(-np.pi,np.pi,Niso)
        lat_iso = np.arccos(np.random.uniform(-1,1,Niso)) - np.pi/2.

        # Cells "occupied" by the isotropic sample in the healpix grid
        iso_cells  = hp.ang2pix(Nside,np.rad2deg(lon_iso),
                np.rad2deg(lat_iso),lonlat=True)

        #healpix_map[iso_cells] = 1
        # Similar to the above
        cells = np.unique(iso_cells,return_counts=True)
        healpix_map[cells[0]] += cells[1]

    if args.raw:
        raw[i,:] = healpix_map

    fractions[i] = Nsrc / Neecr

    if args.raw_only:
        continue

    # Finally: how do we calculate the spectrum?
    if normalize_aps:
        # Calculate the angular power spectrum. Take twice of lmax
        # to make calculations more accurate. This slows down
        # the calculations.
        aps = hp.anafast(healpix_map,lmax=2*lmax)

        # Fix it:
        # Total power of our function: 1/4pi * int |f|^2 dOmega
        # ~ 1/4pi sum ( 4pi/Ncells *f^2 ) = sum(f^2)/Ncells.
        # This is approximately the same as var(healpix_map).
        #total_power = np.sum(healpix_map**2) / Ncells
        total_power = np.var(healpix_map)

        # Sum of Cl coefficients:
        #aps_sum = np.sum(aps)
        aps_sum = np.sum(aps*np.arange(1,4*lmax+2,2))

        # Coefficient of proportion:
        K = total_power / aps_sum

        # "Fixed" spectrum:
        spectrum[i,:] = K * aps[0:lmax+1]

    else:
        ang_data = hp.anafast(healpix_map,lmax=lmax, alm=args.alm)
        if args.alm:
            spectrum[i, :] = ang_data[0]
            if alm is None:
                alm = np.zeros((Nmixed_samples, len(ang_data[1])), dtype=np.complex)
            alm[i,:] = ang_data[1]
        else:
            spectrum[i,:] = ang_data

if alm is None:
    np.savez(outfile1, spectrum=spectrum, fractions=fractions, raw=raw)
else:
    assert(np.sum(np.imag(alm[:,:33])) == 0.)
    np.savez(outfile1, spectrum=spectrum, fractions=fractions, raw=raw, alm=np.real(alm), almi=np.imag(alm[:,33:]))

print('output saved in', outfile1)
