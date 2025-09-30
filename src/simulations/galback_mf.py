import sys
import argparse
import numpy as np
import healpy
from tqdm import tqdm

from crpropa import (
    EeV, kpc, parsec,
    Vector3d, Sphere,
    nucleusId,
    Observer, ObserverSurface, PropagationCK, Candidate,
    MagneticField, ParticleState, ModuleList,
    JF12Field, PlanckJF12bField, JF12FieldSolenoidal, TF17Field,
    PT11Field, #UF23Field, KST24Field
)

from utils import setup_hdf5_file, find_or_create_group, store_results


cline_parser = argparse.ArgumentParser(
    description='E,Z pair sampling',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

def add_arg(*pargs, **kwargs):
    cline_parser.add_argument(*pargs, **kwargs)


add_arg('--mf', type=str,
        help='Magnetic field model used for training (jf | jf_sol | jf_pl | tf | pt | kst | uf)', default='jf')
add_arg('--sample_fname', type=str,
        help='File name with corresponding unique (Z, E), e.g. sample_D3.5_Emin56_100000nuclei_sorted.txt ',
        default='')

add_arg('--Z', type=int, help='Atomic number Z', required=True)
add_arg('--E', type=int, help='Energy in EeV', required=True)

add_arg('--Nside', type=int, help='HEALPix Nside parameter', default=128)
add_arg('--max_step', type=float, help='Maximum step in parsec', default=25)
add_arg('--tolerance', type=float, help='Propagation tolerance', default=1e-4)
add_arg('--galaxy_radius', type=float, help='Galaxy radius in kpc', default=20)
add_arg('--random_seed', type=int, help='Random seed for turbulent fields', default=2**23)
add_arg('--rewrite',  action='store_true',  help='Rewrite if dataset with given parameters already exists')
add_arg('--verbose',  action='store_true',  help='Print coordinates and deflection')

ARGS = cline_parser.parse_args()
NUCLEUS_MAP: dict[int, tuple[int, str]] = {
    1: (1, 'proton'), 2: (4, 'helium'), 3: (7, 'lithium'), 4: (8, 'beryllium'),
    5: (11, 'boron'), 6: (12, 'carbon'), 7: (14, 'nitrogen'), 8: (16, 'oxygen'),
    9: (19, 'fluorine'), 10: (20, 'neon'), 11: (23, 'sodium'), 12: (24, 'magnesium'),
    13: (27, 'aluminium'), 14: (28, 'silicon'), 15: (31, 'phosphorus'), 16: (32, 'sulfur'),
    17: (35, 'chlorine'), 18: (40, 'argon'), 19: (39, 'potassium'), 20: (40, 'calcium'),
    21: (45, 'scandium'), 22: (48, 'titanium'), 23: (51, 'vanadium'), 24: (52, 'chromium'),
    25: (55, 'manganese'), 26: (56, 'iron')
}


def setup_magnetic_field(mf_model: str, mf_params: dict) -> tuple[MagneticField, dict]:
    """Setup magnetic field model with parameters"""

    if mf_model == "jf":
        B = JF12Field()
        mf_params.update({'model': 'JF12', 'striated': 0, 'turbulent': 0})

    elif mf_model == "jf_sol":
        delta_sol, zs_sol = 3, 0.5  # default
        B = JF12FieldSolenoidal(delta_sol * kpc, zs_sol * kpc)
        B.randomStriated(RANDOM_SEED)
        B.randomTurbulent(RANDOM_SEED)
        mf_params.update({'model': 'JF12sol', 'delta_sol': delta_sol, 'zs_sol': zs_sol})

    elif mf_model == "jf_pl":
        B = PlanckJF12bField()
        B.randomStriated(RANDOM_SEED)
        B.randomTurbulent(RANDOM_SEED)
        mf_params.update({'model': 'JF12Planck'})

    elif mf_model == "tf":
        B = TF17Field()
        mf_params.update({'model': 'TF17'})

    elif mf_model == "pt":
        B = PT11Field()
        B.setUseBSS(True)
        B.setUseHalo(True)
        mf_params.update({'model': 'PTKN11'})

    # elif mf_model == "kst":
    #   B = KST24Field()
    #  params.update({'model': 'KST24Field'})

    # elif mf_model == "uf":
    #  B = UF23Field()
    #   params.update({'model': 'UF23Field'})

    else:
        raise ValueError(f"Unsupported magnetic field model: {mf_model}")

    mf_params.update({'random_seed': RANDOM_SEED})

    return B, mf_params


def run_backtracking(B: MagneticField, mf_params: dict, nucleus_params: dict) -> np.ndarray:
    """Run the backtracking simulation"""
    # TODO: add mf parameters

    E, Z, A = nucleus_params['E'], nucleus_params['Z'], nucleus_params['A']
    PID = -nucleusId(A, Z)
    energy = E * EeV

    # Position of the observer
    position = Vector3d(-8.5, 0, 0) * kpc

    # Generate HEALPix grid
    initial_points_number = 12 * Nside * Nside
    initial_coordinates = np.vstack(
        healpy.pixelfunc.pix2ang(Nside, np.arange(initial_points_number),
                                 nest=False, lonlat=False)).transpose()

    results = [] # np.empty(shape=(initial_points_number, 5)) # writing to pre-allocated array is 1.5 times slower

    # Simulation setup
    sim = ModuleList()
    sim.add(PropagationCK(B, tolerance, 0.01 * parsec, max_step * parsec))

    obs = Observer()
    obs.add(ObserverSurface(Sphere(Vector3d(0.), Galaxy_radius * kpc)))
    sim.add(obs)

    points_range = tqdm(range(initial_points_number)) if not ARGS.verbose else range(initial_points_number)

    for i in points_range:
        lat_ini = initial_coordinates[i, 0]
        lon_ini = initial_coordinates[i, 1]

        if lon_ini > np.pi:
            lon_ini = -np.pi + np.mod(lon_ini, np.pi)

        direction = Vector3d()
        direction.setRThetaPhi(1, lat_ini, lon_ini)
        p = ParticleState(PID, energy, position, direction)
        c = Candidate(p)

        sim.run(c)

        d1 = c.current.getDirection()
        lat_res = d1.getTheta()
        lon_res = d1.getPhi()
        deflection = direction.getAngleTo(d1)

        # Convert to degrees and store
        lat_ini_deg, lon_ini_deg, lat_res_deg, lon_res_deg, deflection_deg = \
            np.rad2deg([lat_ini, lon_ini, lat_res, lon_res, deflection])

        if ARGS.verbose:
            print('{:9.3f}{:10.3f}{:9.3f}{:10.3f}{:10.4f}'.
                  format(90 - lat_ini_deg, lon_ini_deg, 90 - lat_res_deg,
                         lon_res_deg, deflection_deg))

        results.append([90 - lat_ini_deg, lon_ini_deg, 90 - lat_res_deg,
                         lon_res_deg, deflection_deg])

        if i % 1000 == 0 and ARGS.verbose:
            print(f"Processed {i}/{initial_points_number} directions")

    return np.array(results, dtype=np.float32)

# TODO: add groups with E/group_0000 for convenience (?)

if __name__ == '__main__':

    Nside = ARGS.Nside
    rewrite = ARGS.rewrite

    max_step = ARGS.max_step
    tolerance = ARGS.tolerance
    Galaxy_radius = ARGS.galaxy_radius
    RANDOM_SEED = ARGS.random_seed

    mf_params = {}  # TO DO : Load config file or sampling

    # Setup magnetic field
    B, params = setup_magnetic_field(ARGS.mf, mf_params=mf_params)

    if ARGS.sample_fname:
        # TODO: read from txt with sorted nuclei
        pass
    else:
        Z = ARGS.Z
        E = ARGS.E

        if Z not in NUCLEUS_MAP:
            raise ValueError(f"Unsupported atomic number Z={Z}")

        print(f"Processing Z={Z}, E={E} EeV")

        A, nucleus = NUCLEUS_MAP[Z]
        nucleus_params = dict(Z=Z, E=E, A=A, nucleus=nucleus)

        import time
        start_time = time.perf_counter()

        with setup_hdf5_file(mf_model=ARGS.mf, Nside=Nside) as h5file:
            group_name, is_new = find_or_create_group(h5file, mf_params, nucleus_params)

            if not is_new and not rewrite:
                sys.exit(f"Group {group_name} already exists. Nothing to be done")

            if not is_new and rewrite:
                print(f"Group {group_name} already exists. Rewriting dataset")

            print("Running backtracking simulation...")

            coordinates = run_backtracking(B, params, nucleus_params=nucleus_params)
            # coordinates = np.random.random((12* Nside * Nside, 5))
            print(f"Storing results in {group_name}")

            store_results(
                h5file, group_name, coordinates,
                mf_params=mf_params, nucleus_params=nucleus_params
            )
            end_time = time.perf_counter()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            print(f"Completed successfully. Results stored in group: {group_name}")

            print(f"Took {elapsed_time:.6f} seconds to execute")

