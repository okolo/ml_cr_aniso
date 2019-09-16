""" The script calculates the angular power spectrum for Nsample samples
of a mixed flux with a fraction coming from a predefined source.
It is to be used then by plot_mean_spectra.py and plot_D2.py 

NB: the EZ_threshold part has not been updated for different methods
of calculating the healpix map and for including N nuclei in a cell.

"""

from __future__ import print_function, division

import sys
import os
import numpy as np
import healpy as hp
#from backports import lzma
import lzma
#______________________________________________________________________

# Total number of EECRs in each sample
Neecr = 500

# fraction of "from-source" EECRs (0,1];
Fsrc = 0.09

# Number of mixed samples (i.e., the sample size)
#Nsample = 10000
Nmixed_samples = 1000

# Emin for which the input sample was generated
Emin = 56   # EeV

#______________________________________________________________________
#source_id = 'NGC253'
source_id = 'CenA'
#source_id = 'M82'
#source_id = 'M87'
#source_id = 'FornaxA'
#______________________________________________________________________

# Calculate the HEALPix map "as is" with 1s in cells with nuclei (or,
# possibly, the number of nuclei) and zeros otherwise OR as the
# relative intensity = (n_i-m_i)/m_i, where n_i is the number of
# nuclei in the ith cell, and m_i being the expected (reference)
# number of nuclei in the cell OR as in the Auger article 1611.06812.
#
# Should be either "asis" (default) or "relins" or "auger"
healpix_map_method = 'asis'

# Should we anyhow normalize the APS?
normalize_aps = 1

# Cut nuclei depending on their rigidity E/Z. No cut is applied if =0:
EZ_threshold = 0

# Less used initial parameters
# healpix grid parameter
Nside = 512

# l_max for the angular power spectrum plot
lmax = 32

# Radius of the vicinity of a source used when making a sample
source_vicinity_radius = 1

# Random seed to reproduce the result if Fsrc<1
random_seed = 2**27

# Size of the initial sample of from-source events. It is used
# in the initial file name and when making a sample of Fsrc*N_EECR
# IT SHOULD NOT BE MODIFIED UNLESS A NEW INPUT FILE IS CREATED
Nini = 10000


#______________________________________________________________________
if Fsrc==1:
    print('Nothing to do with Fsrc=1!')
    sys.exit()

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
print('From-source events: ' + str(Fsrc*100) + '%')
print('Nmixed_samples = ' + str(Nmixed_samples))
print('Healpix map method: ' + healpix_map_method)
if normalize_aps>0:
    print('Normalized spectrum!')
    norm_text = '_norm'
else:
    norm_text = ''

if EZ_threshold>0:
    print('E/Z > ' + str(EZ_threshold) + ' EV')
    EZ_text = '_EZthr'+str(EZ_threshold)
else:
    EZ_text = ''

print('\n')

#______________________________________________________________________
np.random.seed(random_seed)

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
outfile1 = ('aps_'
        + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + EZ_text
        + '_Neecr' + str(Neecr)
        + '_Nsample' + str(Nmixed_samples)
        + '_Fsrc{:02d}'.format(int(round(Fsrc*100)))
        + '_R' + str(source_vicinity_radius) 
        + '_Nside' + str(Nside)
        + norm_text
        + '.txt')

# File for the mean and std of the spectra
#outfile2 = ('data/mixed_spectra/mixed_spectrum_'
outfile2 = ('mean_aps_'
        + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + EZ_text
        + '_Neecr' + str(Neecr)
        + '_Nsample' + str(Nmixed_samples)
        + '_Fsrc{:02d}'.format(int(round(Fsrc*100)))
        + '_R' + str(source_vicinity_radius) 
        + '_Nside' + str(Nside)
        + norm_text
        + '.txt')

# A template for C_l, one line per each isotropic sample
spectrum = np.zeros((Nmixed_samples,lmax+1))

#______________________________________________________________________

# The number of from-source and isotropic-background events
Nsrc = int(np.round(float(Neecr)*Fsrc))
Niso = Neecr - Nsrc

# 1. Read a file produced by src_sample.py:
#    0        1        2        3       4     5  6     7
# lat_ini, lon_ini, lat_res, lon_res, angsep, Z, E, cell_no
#data = np.loadtxt('data/sources/'+infile)
try:
    with lzma.open('data/sources/'+infile,'rt') as f:
        data = np.genfromtxt(f,dtype=float)
        # data = np.loadtxt('data/sources/'+infile)
except IOError:
    print('\n-------> data/sources/' + infile + ' file not found!')
    print('-------> Create one with [convert_]src_sample.py\n')
    sys.exit()

# We need to know the expected flux in a healpix cell if we are
# calculating the relative intensity map or an Auger-like map
#Ncells = 12*Nside**2
Ncells = hp.nside2npix(Nside)
if healpix_map_method=='relins':
    # expected number of Neecr per cell
    expected_flux = float(Neecr)/Ncells
elif healpix_map_method=='auger':
    K = 4*np.pi/Neecr   #/Ncells


# 2. Find non-zero lines, i.e., those with Z>0:
tp = np.arange(0,np.size(data[:,0]))
nonz = tp[data[:,5]>0]
nonz_number = len(nonz)

# For them extract lat, lon_ini (as arrival directions to Earth), and
# possibly E, Z; cell_no can/will be used when calculating the angular
# power spectrum.
# Mark them with "0" to distringuish from samples extracted individually
#lat_src0 = np.deg2rad(data[nonz,0])
#lon_src0 = np.deg2rad(data[nonz,1])

if EZ_threshold>0:
    Z0 = data[nonz,5].astype(int)
    E0 = data[nonz,6].astype(int)

# These are numbers of cells on the healpix grid occupied by the
# nuclei in the infile
healpix_src_cells = data[nonz,7].astype(int)
data = []

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

    if EZ_threshold==0:
        # HEALPix map for from-source events.
        # It is assumed any cell contains at most 1 nucleus.
        # FIX THIS!!!
        if healpix_map_method=='relins':
            # This line must be fixed to include multiplicity
            healpix_map[healpix_src_cell[sample]] = 1/expected_flux - 1
        elif healpix_map_method=='auger':
            healpix_map[healpix_src_cell[sample]] = K - 1
        else:
            # This is a simple way to define the map. It assumes
            # only one EECR gets in a cell. Works OK for Nside=512.
            #healpix_map[src_cells] = 1

            # "cells" is an intermediate variable to simplify the code.
            # It consists of two arrays: [0] is a list of unique cells,
            # [1] is their multiplicity.
            cells = np.unique(src_cells,return_counts=True)
            healpix_map[cells[0]] += cells[1]

    else:
        # _____________________________________________________________
        # This part has not been fixed as the above code.
        # Apply a cut E/Z>EZ_threshold if necessary
        E = E0[sample]
        Z = Z0[sample]

        EZ = np.true_divide(E,Z)
        n_EZ = np.sum(EZ>EZ_threshold)

        #print(('Number of nuclei satisfying E/Z > '+str(EZ_threshold)
        #    +' EV: {:3d}').format(n_EZ))
        if n_EZ==0:
            print('Nothing to do! Skip...')
            continue
        else:
            #lat_src = lat_src[EZ>EZ_threshold]
            #lon_src = lon_src[EZ>EZ_threshold]
            if healpix_map_method=='relins':
                healpix_map[healpix_src_cell[sample[EZ>EZ_threshold]]] = (
                        1/expected_flux - 1 )
            elif healpix_map_method=='auger':
                healpix_map[healpix_src_cell[sample[EZ>EZ_threshold]]]=K-1
            else:
                healpix_map[healpix_src_cell[sample[EZ>EZ_threshold]]] = 1

            #print('Extracted: ' + str(np.sum(healpix_map)))
            #if np.sum(healpix_map) != n_EZ:
            if len(healpix_map[healpix_src_cell[sample[EZ>EZ_threshold]]])!=n_EZ:
                print('Number of selected events mismatch!')
                sys.exit()

    # A sample of events from the isotropic background
    #np.random.seed(iso_random_seed)
    lon_iso = np.random.uniform(-np.pi,np.pi,Niso)
    lat_iso = np.arccos(np.random.uniform(-1,1,Niso)) - np.pi/2.

    # Cells "occupied" by the isotropic sample in the healpix grid
    iso_cells  = hp.ang2pix(Nside,np.rad2deg(lon_iso),
            np.rad2deg(lat_iso),lonlat=True)
    if healpix_map_method=='relins':
        healpix_map[iso_cells] = 1/expected_flux - 1
    if healpix_map_method=='auger':
        healpix_map[iso_cells] = K - 1
    else:
        #healpix_map[iso_cells] = 1
        # Similar to the above
        cells = np.unique(iso_cells,return_counts=True)
        healpix_map[cells[0]] += cells[1]

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
        spectrum[i,:] = hp.anafast(healpix_map,lmax=lmax)

#______________________________________________________________________

# Save spectra of individual samples
#np.savetxt(outfile1,spectrum[0:lmax+1],fmt='%13.5e')
np.savetxt(outfile1,spectrum,fmt='%13.5e')
os.system('xz '+outfile1)

#______________________________________________________________________
# Now, calculate the mean and std of the spectra
print('Calculating mean and std...')
spectrum_mean = np.mean(spectrum,0)
spectrum_std  = np.std(spectrum,0,ddof=1)

# Save parameters of the spectrum
header = '#    mean(C_l)     std(C_l)\n'
with open(outfile2,'w') as d:
    d.write(header)
    for i in np.arange(lmax+1):
        d.write('{:14.5e}{:14.5e}\n'.
            format(spectrum_mean[i],spectrum_std[i]))
        #d.write('{:14.5e}{:14.5e}{:14.5e}{:14.5e}\n'.
        #    format(spectrum_mean[i],spectrum_std[i],
        #        spectrum_min[i],spectrum_max[i]))

dipole = 1.5*np.sqrt(spectrum_mean[1]/np.pi)
print('Dipole amplitude = {:10.4e}'.format(dipole))
#______________________________________________________________________

