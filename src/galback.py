#!/usr/bin/python
# The script is similar to CRPropa's  galactic_backtracking.py
# but needs two input parameters in the command line: Z and E/EeV.
# For example:
# $ python galback.py 1 100 2>&1 >/dev/null &
# to propagate a 100 EeV proton.
# All other initial parameters are 'hardcoded' below. Change them as
# needed

from __future__ import print_function,division
import sys
import os
#import datetime
from crpropa import *
import numpy as np

# _____________________________________________________________________
if len(sys.argv)<3:
    print("usage: python", sys.argv[0], "Z   E/EeV")
    sys.exit(1)

Z = int(sys.argv[1])
E = int(sys.argv[2])

# _____________________________________________________________________
# All other parameters are set below. Fix them as needed.
# The main parameter in healpy_points.py that defines how "precise" is
# the grid.
Nside = 512

# Maximum step in tracking a nucleus, parsec
max_step = 25
tolerance = 1e-4

# Model of the Galactic Magnetic Field
# Pshirkov, Tinyakov, Kronberg, Newton-McGee, ApJ 2011:
# NB: check ASS, BSS, Halo below in the code!
#GMF = "PTKN11"
# Jasson, Farrar, 2012:
GMF = "JF12"
# In case of striated and/or turbulent components:
striated = 1
turbulent = 1
random_seed = 2**27

# Radius of the Milky Way, kpc
Galaxy_radius = 20

# _____________________________________________________________________
# Energy and type of a nucleus to propagate: PID = -nucleusId(A,Z)
# Surely, A and nuclei can be arranged, e.g., as lists, but let's keep
# this for a while:
if Z==1:
    A,nucleus = 1,'proton'
elif Z==2:
    A,nucleus = 4,'helium'
elif Z==3:
    A,nucleus = 7,'lithium'
elif Z==4:
    A,nucleus = 8,'beryllium'
elif Z==5:
    A,nucleus = 11,'boron'
elif Z==6:
    A,nucleus = 12,'carbon'
elif Z==7:
    A,nucleus = 14,'nitrogen'
elif Z==8:
    A,nucleus = 16,'oxygen'
elif Z==9:
    A,nucleus = 19,'fluorine'
elif Z==10:
    A,nucleus = 20,'neon'
elif Z==11:
    A,nucleus = 23,'sodium'
elif Z==12:
    A,nucleus = 24,'magnesium'
elif Z==13:
    A,nucleus = 27,'aluminium'
elif Z==14:
    A,nucleus = 28,'silicon'
elif Z==15:
    A,nucleus = 31,'phosphorus'
elif Z==16:
    A,nucleus = 32,'sufur'
elif Z==17:
    A,nucleus = 35,'chlorine'
elif Z==18:
    A,nucleus = 40,'argon'
elif Z==19:
    A,nucleus = 39,'potassium'
elif Z==20:
    A,nucleus = 40,'calcium'
elif Z==21:
    A,nucleus = 45,'scandium'
elif Z==22:
    A,nucleus = 48,'titanium'
elif Z==23:
    A,nucleus = 51,'vanadium'
elif Z==24:
    A,nucleus = 52,'chromium'
elif Z==25:
    A,nucleus = 55,'manganese'
elif Z==26:
    A,nucleus = 56,'iron'

PID = -nucleusId(A,Z)

# _____________________________________________________________________
# No initial settings below, only code
# Set the magnetic field
if GMF == "JF12":
    B = JF12Field()
    if striated:
        seed_text = ', random seed=' + str(random_seed)
        B.randomStriated(random_seed)
        GMF = GMF + 'S'
    if turbulent:
        seed_text = ', random seed=' + str(random_seed)
        B.randomTurbulent(random_seed)
        GMF = GMF + 'T'
elif GMF=='PTKN11':
    B = PT11Field()
    #B.setUseASS(True)
    B.setUseBSS(True)
    B.setUseHalo(True)
    seed_text = ''
else:
    print('Wrong GMF!')
    sys.exit()

# Name of a 2-column ASCII file with colatitude and longitude
# of the observed particle
input_file  = 'healpy_coordinates_nside' + str(Nside) + '_rad.txt.gz'

# Basename of an output file
output_file = (nucleus + '_{:03d}'.format(E) + 'EeV.txt')

# Header for output_file
output_header = (' PID = ' + str(PID) + ' (' + nucleus
        + '), E = ' + str(E) + ' EeV\n'
        + ' GMF: ' + GMF + seed_text + '. Galaxy radius = '
        + str(Galaxy_radius) + ' kpc. Nside = ' + str(Nside) + '\n'
        + ' Max propagation step = ' + str(max_step) + ' parsec,'
        + ' tolerance = ' + str(tolerance) + '\n'
        + '   lat_ini     lon_ini     lat_res    lon_res   deflection'
        + '\n')

# Write input data to a text file for we could check them immediately
# The file will be overwritten with output data
with open(output_file,'w') as diary:
    diary.write(output_header)

# _____________________________________________________________________
# Backtracking

# Convert energy to EeV (CRPropa)
energy = E * EeV

# Position of the observer
position = Vector3d(-8.5, 0, 0) * kpc

# Load initial data. No check of the validity of the data is performed
try:
    initial_coordinates = np.loadtxt('data/' + input_file)
except IOError:
    # Check if input_file exists, exit if not
    print('\n-------> ' + input_file + ' not found!\n')
    sys.exit()
else:
    initial_file_size = np.shape(initial_coordinates)
    # A strange trick to overcome an error when shape=(2,0)
    # in case of just one point listed in input_file
    if initial_file_size==(2,):
        initial_points_number = 1
    else:
        initial_points_number = initial_file_size[0]

# Just a simple header for on-screen output
print('\n'+ output_header)

# I do not see how one writes an output file line by line yet
# Thus let us create an array that will be saved then
backtracking_results = np.empty(shape=(initial_points_number,5))

# The main cycle over all points in input_file
for i in np.arange(initial_points_number):

    if initial_file_size==(2,):
        lat_ini = initial_coordinates[0]
        lon_ini = initial_coordinates[1]
    else:
        lat_ini = initial_coordinates[i,0]
        lon_ini = initial_coordinates[i,1]

    if lon_ini>np.pi:
        lon_ini = - np.pi + np.mod(lon_ini,np.pi)

    # We can add a check of the validity of data:
    # lat in (0,pi), lon in (0,2pi)

    # It is easier to follow coordinates given in degrees
    #lat_ini_deg = 90 - np.rad2deg(lat_ini)
    #lon_ini_deg = np.rad2deg(lon_ini)
    #print("\n{:4d}. lat = {:6.2f}, lon = {:6.2f}".
    #        format(i+1,lat_deg,lon_deg))

    # Simulation setup
    sim = ModuleList()
    # PropagationCK(Mag. field, tolerance, min step=0.1*kpc,
    # max step=1*Gpc)
    #sim.add(PropagationCK(B, 1e-4, 0.05 * parsec, 1 * parsec))
    sim.add(PropagationCK(B, tolerance, 0.01 * parsec, max_step * parsec))

    obs = Observer()
    # Detects particles upon exiting a sphere
    # NB: Size of the Milky Way
    # https://www.space.com/29270-milky-way-size-larger-than-thought.html
    # https://arxiv.org/abs/1503.00257
    # Deprecated:
    #obs.add(ObserverLargeSphere(Vector3d(0.), Galaxy_radius * kpc))
    obs.add(ObserverSurface(Sphere(Vector3d(0.), Galaxy_radius * kpc)))
    #obs.onDetection(TextOutput(output_file + '.txt', Output.Event3D))
    sim.add(obs)
    #print(sim)

    # Assign initial direction (detected on Earth) and "shoot"
    direction = Vector3d()
    direction.setRThetaPhi(1, lat_ini, lon_ini)
    p = ParticleState(PID, energy, position, direction)
    c = Candidate(p)
    sim.run(c)
    #print(c)

    # Obtain direction at the Milky Way "border"
    d1 = c.current.getDirection()
    lat_res = d1.getTheta()
    lon_res = d1.getPhi()
    deflection = direction.getAngleTo(d1)

    # Convert coordinates and deflections to degrees:
    lat_ini_deg,lon_ini_deg,lat_res_deg,lon_res_deg,deflection_deg \
    = np.rad2deg([lat_ini,lon_ini,lat_res,lon_res,deflection])

    print('{:9.3f}{:10.3f}{:9.3f}{:10.3f}{:10.4f}'.
            format(90-lat_ini_deg,lon_ini_deg,90-lat_res_deg,
                lon_res_deg,deflection_deg))

    backtracking_results[i,:] = [90-lat_ini_deg,lon_ini_deg,
            90-lat_res_deg,lon_res_deg,deflection_deg]

# _____________________________________________________________________
# Finally, save data to output_file
#np.savetxt(output_file+'.txt.gz',backtracking_results,fmt='%10.5f',
#        header=output_header,comments='#')
np.savetxt(output_file,backtracking_results,fmt='%11.5f',
        header=output_header,comments='#')
os.system('xz '+output_file)
# _____________________________________________________________________

