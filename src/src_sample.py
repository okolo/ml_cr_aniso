""" The script reads a file like
`data/sample_D3.5Mpc_Emin56EeV_1000nuclei.txt` obtained with
`psample.py` and then creates a sample of "UHECRs" coming from a
particular source, using "nucleus_NNNEeV.txt.xz" obtained with
galback.py and located in a directory that corresponds to the
selected Nside and the GMF model (e.g., in data/jf/512/).

Input:
    * source_id='CenA' -- the "name" of an UHECR source
    * Nini=1000 -- the number of nuclei in the initial sample
      (obtained with `psample.py`)
    * Emin=56 [EeV]
    * Nside=512 -- the HEALPix parameter
    * source_vicinity_radius = 1 [deg]
    * a file `data/sample_D..._sorted.txt` prepared with `psample.py`
    * GMF = 'PTKN11'|'JF12ST' -- the GMF model

Output:
    * a file like
      `src_sample_CenA_D3.5_Emin56_N1000_R1_Nside512.txt.xz`
      to be placed in `data/GMF/sources/` where GMF=[jf|pt]
      depending on the chosen GMF model.

An output file contains 8 columns:
    * arrival direction at Earth (lat_ini, lon_ini), degrees;
    * arrival direction at the boundary of the Galaxy (lat_res,
      lon_res); these will be similar to each other because of the
      small radius of the vicinity of the direction to the given source;
    * angular separation between (lat_res, lon_res) and the direction to
      the source;
    * (Z, E) pair;
    * number of the cell that contains (lat_res, lon_res) on the healpix
      grid.

Notation (lat_ini, lon_ini) and (lat_res, lon_res) originates from the
fact that we perform backtracking of (anti-)nuclei, so that their
initial position is at Earth. Perhaps, they should be renamed to, e.g.,
(lat_earth, lon_earth) and (lat_gal, lon_gal).

The script reads a file like `data/sample_D3.5Mpc..._sorted.txt`
obtained with `psample.py`, thus *Nini* and *Emin* must be exactly
the same as used in `psample.py` for the distance to the given
source (see *source_id*).

It then searches for a sample of Nini events in a given
`source_vicinity_radius` from files `nucleus_NNNEeV.txt.xz` obtained
with `galback.py`.  An output file is called `src_sample_ID_....txt`,
where ID is the *source_id*.  It must be placed in `data/sources/`.  The
sample can then be used to plot a figure and/or create a mix of samples
with an isotropic background.  

*NB: an output sample does NOT necessarily contain Nini nuclei since
there may be less "suitable" (Z,E) pairs coming form the vicinity of the
source on a given healpix' grid! Use greater Nside if necessary or
increase the radius of the vicinity*

`convert_src_sample.py` can be used to convert data from
`data/GMF/sources/src_sample_...NsideXXX.txt.xz` file to data (and file)
calculated for another Nside.

An output file is called src_sample_ID....txt.xz and is placed
in data/GMF/sources/
"""


import sys
import numpy as np
from PyAstronomy.pyasl import getAngDist
import argparse

import lzma
import healpy as hp
import os

#______________________________________________________________________
#source_id = 'NGC253'
source_id = 'CenA'
#source_id = 'M82'
#source_id = 'M87'
#source_id = 'FornaxA'
#______________________________________________________________________

# Total number of EECRs to be found in the given vicinity of the source.
# This is the same number as used in psample.py
Nini = 10000

# Minimum energy used to create a sample, EeV:
Emin = 56

# Healpix grid used in simulations:
#Nside = 512
Nside = 256

# Radius of the source neighborhood, deg
source_vicinity_radius = 1

shiftA = 1.0

# Model of the Galactic Magnetic Field
#GMF = 'jf'
GMF = 'pt'

#______________________________________________________________________
# No settings below

source_data = {
    # Name : [source_lon, source_lat, D_src]
    'M82': [141.4095,40.5670,'3.5'],
    'CenA': [309.5159,19.4173,'3.5'],
    'NGC253': [97.3638,-87.9645,'3.5'],
    'NGC6946': [95.71873,11.6729,'6.0'],
    'M87': [283.7777,74.4912, '18.5'],
    'FornaxA': [240.1627,-56.6898,'20.0']
}

def get_source_data(source_id):
    if source_id in source_data:
        return tuple(source_data[source_id])
    else:
        raise ValueError('Unknown source!')

cline_parser = argparse.ArgumentParser(description='E,Z pair sampling',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('--source_id', type=str, help='one of ' + ' , '.join(source_data.keys()), default=source_id)
add_arg('--GMF', type=str, help='pt or jf', default=GMF)
add_arg('--Emin', type=float, help='Emin in EeV', default=Emin)
add_arg('--Nini', type=int, help='Number of cosmic ray nuclei to form a sample', default=Nini)
add_arg('--Nside', type=int, help='Nside for output healpix grid', default=Nside)
add_arg('--shiftA', type=float, help='A factor to shift and atomic mass by', default=shiftA)


args = cline_parser.parse_args()
source_id = args.source_id
GMF = args.GMF
Emin = args.Emin
Nini = args.Nini
Nside = args.Nside
shiftA = args.shiftA

source_lon, source_lat, D_src = get_source_data(source_id)

#______________________________________________________________________
# A file in spectra_1s/

gmf_dir = GMF + '/'

# File with the spectrum that was "propagated" with CRPropa
infile = ('data/sample_D'+D_src+'_Emin'+str(Emin)+'_'
        + str(Nini))
if shiftA != 1.0:
    infile += '_shift' + str(shiftA)

infile += 'nuclei_sorted.txt'

outfile = ('src_sample_' + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + '_N' + str(Nini) + '_R'
        + str(source_vicinity_radius) + '_Nside' + str(Nside))
if shiftA != 1.0:
    outfile += '_shift' + str(shiftA)

# A template for the output file
# lat, lon (ini); lat, lon (res); angsep; Z, E, # in the healpix map
outdata = np.zeros([Nini,8],dtype=float)

# Read the file and extract data
data = np.loadtxt(infile,dtype=int)

# Z, E/EeV, multiplicity of the nuclei
Z = data[:,0]
E = data[:,1]
M = data[:,2]

# Number of nucleus_energy.txt.xz files to be proceeded
N_files = len(Z)
#N_files = 2

# A list of nuclei to prepare the name of a file to read on the fly
nuclei = ['proton','helium','lithium','beryllium','boron','carbon',
        'nitrogen','oxygen','fluorine','neon','sodium','magnesium',
        'aluminium','silicon','phosphorus','sufur','chlorine','argon',
        'potassium','calcium','scandium','titanium','vanadium',
        'chromium','manganese','iron']

# A counter of EECRs added to the output file (result)
k = 0

# A counter for missing (Z,E) pairs
err_no = 0

# Let us check first that all necessary files exist
for i in np.arange(N_files):
    nucleus = nuclei[Z[i]-1]
    filename = nucleus + '_{:03d}EeV.txt.xz'.format(E[i])
    file2load = ('data/'+gmf_dir+str(Nside)+'/'+filename)

    print(nucleus + ' (Z = ' +str(Z[i])+ ') ' + str(E[i]) + ' EeV')

    # Check file
    if os.path.isfile(file2load):
        print('OK')
    else:
        # This is to be copy-pasted in run_galback.py or
        # a similar script
        print(('-----> MISSING: '+ nucleus
                + '\t(Z,E) = {:2d}\t{:3d}\t{:3d} '
                + 'nuclei needed')
                .format(Z[i],E[i],M[i]))
        err_no += M[i]

if err_no>0:
    print('Files for some (Z,E) are missing.')
    print('{:} nuclei will not be included in the output'
            .format(err_no))
    # Normally, we should exit at this point but let us allow
    # going forward in case the number of missing nuclei is much
    # less than the size of the sample.
    #sys.exit()

print('-------------------------------------\n')

# _____________________________________________________________________
# A cycle over all nuclei listed in infile
for i in np.arange(N_files):
    nucleus = nuclei[Z[i]-1]
    filename = nucleus + '_{:03d}EeV.txt.xz'.format(E[i])

    print(nucleus + ' (Z = ' +str(Z[i])+ ') ' + str(E[i]) + ' EeV')

    try:
        file2load = ('data/'+gmf_dir+str(Nside)+'/'+filename)
        with lzma.open(file2load,'rt') as f:
            summary = np.genfromtxt(f,dtype=float)

        # Coordinates observed at Earth
        #lat_earth_deg = summary[:,0]
        #lon_earth_deg = summary[:,1]
        # Coordinates at the boundary of the Galaxy
        lat_gal_deg = summary[:,2]
        lon_gal_deg = summary[:,3]
        # We need angular separtion between the source and
        # arrival directions at the Galaxy boundary
        ang_sep = getAngDist(source_lon, source_lat,
                lon_gal_deg, lat_gal_deg)

        # Find EECRs that fit in the given source vicinity at
        # the Galaxy boundary, if any
        close_dirs = np.asarray(
                np.where(ang_sep<=source_vicinity_radius+0.01))
        N_close = np.size(close_dirs)

        if N_close:
            print('{:4d} close EECRs'.format(N_close))
            print('To be selected: ' + str(M[i]) + '\n')

            close_dirs = np.reshape(close_dirs,N_close)

            # This is the 1st version: arrival directions are chosen
            # w/0 replacement. Due to this, we sometimes lack a number
            # of arrival directions/nuclei in the output sample
            #sample_size = np.min([M[i],N_close])
            #sample = np.random.choice(close_dirs,size=sample_size,
            #        replace=False)

            # Version 2. Allow replacement in case there are not enough
            # close arrival directions. This will give an output closer
            # to the given spectrum
            sample_size = M[i]
            if M[i]<=N_close:
                sample = np.random.choice(close_dirs,size=sample_size,
                        replace=False)
            else:
                sample = np.random.choice(close_dirs,size=sample_size,
                        replace=True)

            # OK, this is for the output file
            for j in sample:
                outdata[k,0:4] = summary[j,0:4]
                outdata[k,4] = ang_sep[j]
                outdata[k,5] = Z[i]
                outdata[k,6] = E[i]
                outdata[k,7] = j
                k += 1

            summary = []

        else:
            print('No close EECRs')
            print('To be selected: ' + str(M[i]) + '\n')

    # File is not available
    except IOError:
        print('\n-------> ' + filename + ' file not found!\n')
        #sys.exit()

#______________________________________________________________________
# Write down the output file and xz it

header = ('#   lat_ini    lon_ini    lat_res    lon_res     angsep   '
          'Z   E   cell_no\n')

with open('data/' + gmf_dir + 'sources/' + outfile + '.txt','w') as d:
    d.write(header)
    for i in np.arange(Nini):
        d.write('{:11.5f}{:11.5f}{:11.5f}{:11.5f}{:11.5f}{:4d}{:5d}{:9d}\n'.
                format(outdata[i,0],outdata[i,1],
                   outdata[i,2],outdata[i,3],
                   outdata[i,4],int(outdata[i,5]),
                   int(outdata[i,6]),int(outdata[i,7])))

os.system('xz data/' + gmf_dir + 'sources/' + outfile + '.txt')

#______________________________________________________________________
