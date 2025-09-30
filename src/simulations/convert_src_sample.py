"""
The script takes an src_sample_...NsideXXX.txt.xz file
from data/sources/ and converts it to a similar file for another Nside.

"""

#from __future__ import print_function, division

import sys
import os
import numpy as np
import healpy as hp
import lzma
import argparse

#______________________________________________________________________

# Nside in the initial file
Nside_ini = 256

#source_id = 'NGC253'
source_id = 'CenA'
#source_id = 'M82'
#source_id = 'M87'
#source_id = 'FornaxA'
#______________________________________________________________________

# Emin for which the input sample was generated
Emin = 56   # EeV

# Less used initial parameters
# healpix grid parameter
Nside = 128

# Radius of the vicinity of a source used when making a sample
source_vicinity_radius = '1'

# data folder for selected galactic magnetic field model
gmf_dir = 'jf'

# Size of the initial sample of from-source events. It is used
# in the initial file name and when making a sample of src_frac*N_EECR
# IT SHOULD NOT BE MODIFIED UNLESS A NEW INPUT FILE IS CREATED
Nini = 10000

# Mass composition shift used for map generation
shiftA = '1'

#_________________________  parsing command line

cline_parser = argparse.ArgumentParser(description='src_sample_XXX.txt.xz to another (lower) Nside ',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


add_arg('--Nside_ini', type=int, help='Source Nside value', default=Nside_ini)
add_arg('--Nside', type=int, help='Target Nside value', default=Nside)
add_arg('--Emin', type=int, help='Emin in EeV for which the input sample was generated', default=Emin)
add_arg('--source_vicinity_radius', type=str, help='Radius of the vicinity of a source used when making a sample',
        default=source_vicinity_radius)
add_arg('--mf', type=str, help='data folder for selected galactic magnetic field model (pt, jf, etc.)',
        default=gmf_dir)
add_arg('--Nini', type=int, help='Size of the initial sample of from-source events.', default=Nini)
add_arg('--shiftA', type=str, help='A factor to shift and atomic mass by', default=shiftA)
add_arg('--source', type=str, help='Source name', default=source_id)

args = cline_parser.parse_args()
Nside_ini = args.Nside_ini
Emin = args.Emin
source_vicinity_radius = args.source_vicinity_radius
gmf_dir = args.mf
Nini = args.Nini
shiftA = float(args.shiftA)
source_id = args.source
Nside = args.Nside

#______________________________________________________________________

if Nside==Nside_ini:
    print("Nside=Nside_ini, nothing to be done!")
    sys.exit()

#______________________________________________________________________
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

if not gmf_dir.endswith('/'):
    gmf_dir += '/'

dir_prefix = 'data/'+gmf_dir+'sources/'

# Input file name; the file must be prepared with src_sample.py
# It provides data for making a sample of from-source events
infile = ('src_sample_' + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + '_N' + str(Nini)
        + '_R' + str(source_vicinity_radius) 
        + '_Nside' + str(Nside_ini))

if shiftA != 1.0:
    infile += '_shift' + args.shiftA

infile += '.txt.xz'

# File for individual spectra
outfile = ('src_sample_' + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + '_N' + str(Nini)
        + '_R' + str(source_vicinity_radius) 
        + '_Nside' + str(Nside))

if shiftA != 1.0:
    outfile += '_shift' + args.shiftA

outfile += '.txt'

#______________________________________________________________________

# 1. Read a file produced by src_sample.py:
#    0        1        2        3       4     5  6     7
# lat_ini, lon_ini, lat_res, lon_res, angsep, Z, E, cell_no
#data = np.loadtxt('data/sources/'+infile)
try:
    with lzma.open(dir_prefix+infile,'rt') as f:
        data = np.genfromtxt(f,dtype=float)
except IOError:
    print('\n-------> ' +dir_prefix+infile+' file not found!\n')
    sys.exit()

# 2. Find non-zero lines, i.e., those with Z>0:
tp = np.arange(0,np.size(data[:,0]))
# These are indices of "nonzero nuclei" in the initial sample
nonz = tp[data[:,5]>0]

# Extract lat_ini, lon_ini (as arrival directions to Earth), and
# possibly E, Z; cell_no can/will be used when calculating the angular
# power spectrum.
lat_arr = np.deg2rad(data[nonz,0])
lon_arr = np.deg2rad(data[nonz,1])

# These are numbers of cells on the healpix grid.
healpix_cells  = hp.ang2pix(Nside,np.rad2deg(lon_arr),
        np.rad2deg(lat_arr),lonlat=True)

# Now we only have to write down the outfile and xz it.

header = ('#   lat_ini    lon_ini    lat_res    lon_res     angsep   '
          'Z   E   cell_no\n')

with open(dir_prefix+outfile,'w') as d:
    d.write(header)
    for i in np.arange(len(nonz)):
        d.write('{:11.5f}{:11.5f}{:11.5f}{:11.5f}{:11.5f}{:4d}{:5d}{:9d}\n'.
                format(data[i,0],data[i,1],
                   data[i,2],data[i,3],
                   data[i,4],int(data[i,5]),
                   int(data[i,6]),int(healpix_cells[i])))

os.system('xz '+dir_prefix+outfile)
# EOF
