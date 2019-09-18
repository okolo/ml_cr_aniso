""" The script plots a simple plot with Neecr arrival directions
with Fsrc%% of them coming from the given source_id.

The script needs an input file in data/GMF/sources prepared with
src_sample.py for that particular source_id, Emin, Nside and GMF.
"""

#from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import lzma

# 0. Initial data:

# Neecr (size of a sample to be plotted);
Neecr = 500

# fraction of "from-source" EECRs (0,1];
Fsrc = 9/100.

# GMF model!!!
GMF = 'PTKN11'
#GMF = 'JF12ST'

#______________________________________________________________________
#                           100     200     300     400     500
#source_id = 'NGC253'   #    17      12      10      8       7
source_id = 'CenA'     #    21      14      12      10      9
#source_id = 'M82'      #    26      18      14      12      11
#source_id = 'M87'      #    29      20      16      14      12
source_id = 'FornaxA'  #    19      13      11      9       8
#______________________________________________________________________

#______________________________________________________________________

# Cut nuclei depending on E/Z. No cut is applied if =0:
EZ_threshold = 0

# optionally (to be used in a plot title): Nside, Emin, D_src.
Nside = 256
Emin  = 56  # EeV

# Radius of the vicinity of a source used when making a sample
source_vicinity_radius = 1

# Random seed to reproduce the result if Fsrc<1
random_seed = 2**26
iso_random_seed = 2**26

# Size of the initial sample of from-source events. It is used
# in the initial file name and when making a sample of Fsrc*Neecr
Nini = 10000

# Should we save a sample in a separate (txt) file?
save_sample = 0

# Do we need a pdf plot w/o a title?
plot_pdf = 0

#______________________________________________________________________
if source_id=='M82':
    source_lon = 141.4095
    source_lat = 40.5670
    D_src = '3.5'    # Mpc
    legend_loc = 'upper left'
elif source_id=='CenA':
    source_lon = 309.5159
    source_lat = 19.4173
    D_src = '3.5'    # Mpc
    legend_loc ='upper right'
elif source_id=='NGC253':
    source_lon = 97.3638
    source_lat = -87.9645
    D_src = '3.5'    # Mpc
    legend_loc = 'upper right'
elif source_id=='NGC6946':
    source_lon = 95.71873
    source_lat = 11.6729
    D_src = '6.0'
    legend_loc ='upper left'
elif source_id=='M87':
    source_lon = 283.7777
    source_lat = 74.4912
    D_src = '18.5'    # Mpc
    legend_loc = 'upper right'
elif source_id=='FornaxA':
    source_lon = 240.1627
    source_lat = -56.6898
    D_src = '20.0'    # Mpc
    legend_loc = 'upper right'
else:
    print('\nUnknown source!')
    sys.exit()

#______________________________________________________________________
if EZ_threshold>0:
    EZ_name = '_EZthr'+str(EZ_threshold)
else:
    EZ_name = ''
    EZ_text = ''

if GMF=='PTKN11':
    gmf_dir = 'pt/'
else:
    gmf_dir = 'jf/'

#______________________________________________________________________
# input file name;

infile = ('src_sample_' + source_id + '_D' + D_src
        + '_Emin' + str(Emin)
        + '_N' + str(Nini)
        + '_R' + str(source_vicinity_radius) 
        + '_Nside' + str(Nside)
        + '.txt.xz')

# Perhaps, there is sense to add Nini since different Nini result in
# slightly different plots
figname = (source_id + '_sample'
        #+ '_D' + D_src
        + '_Emin' + str(Emin)
        + EZ_name
        + '_R' + str(source_vicinity_radius)
        + '_Neecr' + str(Neecr)
        + '_Fsrc{:02d}'.format(int(round(Fsrc*100)))
        + '_Nside' + str(Nside)
        )

# A list of nuclei to prepare the name of a file to read on the fly
nuclei = ['proton','helium','lithium','beryllium','boron','carbon',
        'nitrogen','oxygen','fluorine','neon','sodium','magnesium',
        'aluminium','silicon','phosphorus','sufur','chlorine','argon',
        'potassium','calcium','scandium','titanium','vanadium',
        'chromium','manganese','iron']

# _____________________________________________________________________
# 1. Read a file produced by src_sample.py:
#    0        1        2        3       4     5  6     7
# lat_ini, lon_ini, lat_res, lon_res, angsep, Z, E, cell_no
try:
    with lzma.open('data/'+gmf_dir+'sources/'+infile,'rt') as f:
        data = np.genfromtxt(f,dtype=float)
        # data = np.loadtxt('data/sources/'+infile)
except IOError:
    print('\n-------> ' + infile + ' file not found!\n')
    sys.exit()


# _____________________________________________________________________
# 2. Find non-zero lines, i.e., those with Z>0:
#nonz = np.asarray(np.where(data[:,5]>2*sys.float_info.epsilon))
#nonz = (data[:,5]>0).nonzero()
tp = np.arange(0,np.size(data[:,0]))
nonz = tp[data[:,5]>0]
#nonz = np.reshape(nonz,(np.size(nonz),0))

# For them extract lat, lon_ini (as arrival directions to Earth),
# and Z (possibly E) to be used in a plot legend. cell_no can/will
# be used when calculating the angular power spectrum.
lat_src = np.deg2rad(data[nonz,0])
lon_src = np.deg2rad(data[nonz,1])
Z = data[nonz,5].astype(int)
E = data[nonz,6].astype(int)

# These are numbers of cells on the healpix grid
healpix_cell = data[nonz,7].astype(int)
data = []


# _____________________________________________________________________
# 3. If a fraction of from-source EECRs =1, make a plot;
#   otherwise:
#   - calculate the number of "background" (Nb) and the number of
#     from-source (Ns) EECRs;
#   - extract Ns from-source events;
#   - create a sample of Nb events of the isotropic background;

if Fsrc<1:

    # The number of from-source and isotropic background events
    N_src = int(np.round(float(Neecr)*Fsrc))
    N_iso = Neecr - N_src

    # A sample of events from the isotropic background
    np.random.seed(iso_random_seed)
    lon_iso = np.random.uniform(-np.pi,np.pi,N_iso)
    lat_iso = np.arccos(np.random.uniform(-1,1,N_iso)) - np.pi/2.

else:
    N_src = Neecr

# _____________________________________________________________________
# Create a sample we need according to Neecr and Fsrc
np.random.seed(random_seed)
# A sample of from-source events:
if np.size(nonz)>N_src:
    sample = np.random.choice(np.size(nonz),N_src,replace=False)
elif np.size(nonz)<N_src:
    print('The input sample of from-source events is small:')
    print('We need N_src={:4d} but only have {:4d}'.
            format(N_src,np.size(nonz)))
    sample = np.random.choice(np.size(nonz),N_src,replace=True)
else:
    sample = nonz

lat_src = lat_src[sample]
lon_src = lon_src[sample]
Z = Z[sample]
E = E[sample]

# _____________________________________________________________________
# Apply a cut E/Z>EZ_threshold if necessary
if EZ_threshold>0:
    EZ = np.true_divide(E,Z)
    n_EZ = np.sum(EZ>EZ_threshold)

    print('\nInitial number of nuclei {:3d}'.format(len(E)))
    print(('Number of nuclei satisfying E/Z > '+str(EZ_threshold)
        +' EV: {:3d}').format(n_EZ))
    if n_EZ==0:
        print('Nothing to do! Exit...')
        sys.exit()
    else:
        lat_src = lat_src[EZ>EZ_threshold]
        lon_src = lon_src[EZ>EZ_threshold]
        Z = Z[EZ>EZ_threshold]
        E = E[EZ>EZ_threshold]

        EZ_text = ('Cut: E/Z>'+str(EZ_threshold)+' EV. '
                + str(n_EZ) + ' EECRs remain')

# _____________________________________________________________________
plot_header = ('Sample: ' + str(Neecr)+' EECRs, min(E)='
        + str(Emin) + ' EeV. '
        + 'From the source: {:3.0f}%. '.format(Fsrc*100)
        + EZ_text
        + ' GMF: ' + GMF
        #+ 'Source: ' + source_id + '('+D_src + ' Mpc), '
        )

# _____________________________________________________________________
# Make a plot with the background events possibly plotted in black,
# and from-source events in colours w.r.t. their Z; a colourmap can
# probably be used to show energies of these events.

# Let us find the number of unique Z to use in the legend
unique_Z = np.unique(Z)
N_unique_Z = len(unique_Z)

# A trick for nuclei colours
start_cm = 0.0
stop_cm  = 1.0
Ncolors  = len(nuclei)

cm_subsection = np.linspace(start_cm, stop_cm, Ncolors) 
#colors = [ plt.cm.viridis(x) for x in cm_subsection ]
#colors = [ plt.cm.tab20c(x) for x in cm_subsection ]
#colors = [ plt.cm.hsv(x) for x in cm_subsection ]
#colors = [ plt.cm.Spectral(x) for x in cm_subsection ]
colors = [ plt.cm.gist_rainbow(x) for x in cm_subsection ]

fig=plt.figure(figsize=(10,7))
plt.subplot(111, projection = 'mollweide')

# A template for markers; one more for the background if any
if (Fsrc<1) and (Neecr<=1000):
    marker = ['a']*(N_unique_Z+1)
    # Plot background events
    marker[-1] = plt.scatter(lon_iso, lat_iso,
            marker='o', s=10,
            facecolor='1',edgecolor='k',
            label = 'background')
else:
    marker = ['a']*(N_unique_Z)

# Plot from-source EECRs
for i in np.arange(0,N_unique_Z):
    nucleus = nuclei[unique_Z[i]-1]
    #print(nucleus)

    this_Z  = np.asarray(np.where(Z==unique_Z[i]))
    #symbol_size = 50*np.round(E[this_Z]/float(Emin))
    symbol_size = 50
    marker[i] = plt.scatter(lon_src[this_Z[0]], lat_src[this_Z[0]],
            marker='o', s=symbol_size,
            facecolor=colors[unique_Z[i]-1],
            edgecolor='k',
            label=nucleus)

    # This is for the case we decide to scale symbols according to
    # the energy of arriving nuclei
    #symbol_size = 50*np.round(pow(E[this_Z]/float(Emin),1.1))
    for j in np.arange(len(this_Z)):
        #print(E[this_Z],np.rad2deg(lon_src[this_Z[j]]),
        #        np.rad2deg(lat_src[this_Z[j]]))
        plt.scatter(lon_src[this_Z[j]], lat_src[this_Z[j]],
                marker='o',
                #s=symbol_size[j],
                s=symbol_size,
                facecolor=colors[unique_Z[i]-1],
                edgecolor='k')

# That's all. Let us plot the source, legend and title
# This is only needed for plotting but let it be for calcs, too
if source_lon>180:
    source_lon -= 360

src = plt.scatter(np.deg2rad(source_lon), np.deg2rad(source_lat),
        marker='*', c='red', s=150,label=source_id)

plt.grid(True)

# Finally, let us make a good legend
legend_text = [src]
for t in np.arange(len(marker)):
    legend_text += [marker[t]]

# Plot the source position
plt.legend(handles=legend_text,
    scatterpoints=1, loc=legend_loc, ncol=2, fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.rcParams.update({'font.size': 14})
if plot_pdf>0:
    fig.savefig(figname+'.pdf')

plt.title(plot_header)
fig.savefig(figname+'.png')
plt.show()
#plt.close()

# EOF
