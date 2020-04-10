from __future__ import print_function
import numpy as np
import argparse

# _____________________________________________________________________
# A file in spectra_1s/
distance = '3.5'

# Minimum energy in the spectrum
Emin = '56'   # EeV

# (Initial) Number of cosmic ray nuclei to form a sample
Nini = 100000

# Random seed to reproduce the spectrum if needed
random_seed = 2**26

# a factor to shift and atomic mass by
shiftA = 1.

# _____________________________________________________________________
# No settings below

cline_parser = argparse.ArgumentParser(description='E,Z pair sampling',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)

add_arg('--distance', type=str, help='distance in Mpc to source (corresponding file should exist in spectra_1s/)', default=distance)
add_arg('--Emin', type=str, help='Emin in EeV', default=Emin)  # using string type to make sure output file name has Emin parameter in the same format (fix rounding issue)
add_arg('--Nini', type=int, help='Number of cosmic ray nuclei to form a sample', default=Nini)
add_arg('--shiftA', type=str, help='A factor to shift and atomic mass by (if negative use mono composition with A=-shiftA)', default=shiftA)  # using string type to fix output file naming

args = cline_parser.parse_args()
distance = args.distance
Emin = float(args.Emin)
Nini = args.Nini
shiftA = float(args.shiftA)
if shiftA < 0:
    shiftA = int(shiftA)

spec = np.loadtxt('spectra_1s/' + distance + '.gz')

output_file = 'sample_D' + distance + '_Emin' + args.Emin + '_' + str(Nini)
if shiftA != 1.:
    output_file += '_shift' + args.shiftA
output_file += 'nuclei.txt'


# write a diary to be used in galactic backtracking
with open(output_file,'w') as d:
    d.write('# Distance to source: ' + distance + ' Mpc\n')
    d.write('# Minimum energy: ' + str(Emin) + ' EeV\n')
    d.write('# Number of nuclei: ' + str(Nini) + '\n')
    d.write('# Random seed: ' + str(random_seed) + '\n')
    d.write('# Produced with psample_integer.py\n')
    if shiftA != 1:
        if shiftA > 0:
            d.write('# shift in A: ' + str(shiftA) + '\n')
        else:
            d.write('# mono composition A =' + str(shiftA) + '\n')
    d.write('#  Z   E,EeV\n')

# _____________________________________________________________________
Emin = float(Emin)*1e18

# cut lower energies
spec = spec[spec[:,0]>=Emin, :]

# E column
E = np.reshape(spec[:,0],(spec.shape[0], 1))

# convert J*E^2 to J*E (probability is proportional to J*E in log scale)
spec[:,1:] /= E

# calculate total probability for each nucleus
probabA=np.sum(spec[:,1:], axis=0)
# normalize probability
probabA = probabA / np.sum(probabA)

stable_isotope_charge = [1,1,2,2,3,3,4,4,4,5,5,6,6,7,7,8,8,8,9,
        10,10,10,11,12,12,12,13,14,14,14,15,16,16,16,17,16,17,
        18,19,20,20,20,20,20,21,22,22,22,22,22,23,24,24,24,25,26]

#print('   Z    A    E, eV')
print('   Z    E, EeV')

np.random.seed(random_seed)
# choose nucleus
for A in np.random.choice(len(probabA),size=Nini,replace=True,p=probabA):
    probab = spec[:,1+A]
    # normalize probability for the nucleus chosen
    probab /= np.sum(probab)
    # choose particle energy
    energy = np.random.choice(E[:,0], p=probab)
    #print('{:4d}{:5.0f}{:10.2e}'.format(
    #    stable_isotope_charge[A],A+1,energy))
    print('{:4d}{:6.0f}'.format(
        stable_isotope_charge[A], np.round(energy/1e18)))

    if shiftA < 0:
        A = -shiftA - 1  # A is zero-based index
        assert 0 <= A <= 55
    else:
        A = int(np.round(shiftA*A))  # shift atomic mass by constant factor (used for composition dependence test)
        A = min(A, 55)  # make sure A is not heavier than Fe

    with open(output_file,'a') as d:
        d.write('{:4d}{:6.0f}\n'.format(
            stable_isotope_charge[A], np.round(energy/1e18)))

