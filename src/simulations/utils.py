"""
Managing h5py files with deflections maps:
creating, writing, checking groups
"""
# GMF - Galactic Magnetic Field

import h5py
import numpy as np
from pathlib import Path

DATA_PATH = Path('../data/')


def setup_hdf5_file(mf_model: str, Nside: int) -> h5py.File:
    """
    Create or open HDF5 file and return file handle.

    Parameters
    ----------
    mf_model: gmf model jf | jf_sol | jf_pl | tf | pt | kst | uf
    Nside: HEALPix Nside parameter

    Returns
    ----------
    h5py File handle
    """
    hdf5_path = DATA_PATH / f"{mf_model}_Nside{Nside}_deflection_maps.h5"

    if not hdf5_path.parent.exists():
        hdf5_path.parent.mkdir(parents=True)

    if not hdf5_path.exists():
        with h5py.File(hdf5_path, 'w') as f:
            f.create_group('metadata')
            f['metadata'].create_dataset('parameter_table', dtype=h5py.special_dtype(vlen=str))

    return h5py.File(hdf5_path, 'a')



def find_or_create_group(h5file: h5py.File, mf_params: dict, nucleus_params: dict) -> tuple[str, bool]:
    """
    Find existing group or create new one for given parameters

    Parameters
    ----------
    h5file: file containing deflection maps obtained with given GMF model
    mf_params: parameters of the GMF model
    nucleus_params: dict with keys nucleus charge Z, energy E

    Returns
    ----------
        group_name[str]: name of created/existing group of parameters
        status[bool]: whether a new group is created
    """
    param_sig = create_param_signature(nucleus_params, mf_params)
    metadata = h5file['metadata']

    if metadata['parameter_table'].shape is not None:
        param_table = metadata['parameter_table'][:]
        param_table = np.array([s.decode('utf-8') for s in param_table])

        if param_sig in param_table:
            group_idx = np.where(param_table == param_sig)[0][0]
            group_name = f"group_{group_idx:04d}"
            return group_name, False

        else:
            param_table = list(metadata['parameter_table'][:])
            param_table.append(param_sig)

            del metadata['parameter_table']
            metadata.create_dataset(
                'parameter_table',
                data=np.array(param_table, dtype=object),
                dtype=h5py.special_dtype(vlen=str))

            group_idx = len(param_table) - 1
            group_name = f"group_{group_idx:04d}"
            return group_name, True

    del metadata['parameter_table']
    metadata.create_dataset(
        'parameter_table',
        data=np.array([param_sig], dtype=object),
        dtype=h5py.special_dtype(vlen=str)
    )

    group_idx = 0
    group_name = f"group_{group_idx:04d}"
    return group_name, True


def store_results(
        h5file: h5py.File, group_path: str,
        results: np.ndarray,
        mf_params: dict,
        nucleus_params: dict
) -> None:
    """
    Store results in HDF5 group

    Parameters
    ----------
    h5file:  file containing deflection maps obtained with given GMF model
    group_path: name of the group
    results: array with corresponding [lat_ini, lon_ini, lat_res, lon_res, deflection]
    mf_params: parameters of the GMF model
    nucleus_params: parameters of the nucleus
    """

    if group_path in h5file:
        del h5file[group_path] # just in case something went wrong

    group = h5file.create_group(group_path)

    # let it be for a while
    group.attrs.update(mf_params)
    group.attrs.update(nucleus_params)

    group.create_dataset('coordinates', data=results, compression='gzip')


def create_param_signature(nucleus_params: dict, mf_params: dict) -> str:
    nucleus_str = f"Z{nucleus_params['Z']}_E{nucleus_params['E']}_A{nucleus_params['A']}"

    # Sort mf_params by key for consistent ordering
    mf_str = "_".join([f"{k}={v}" for k, v in sorted(mf_params.items())])

    return f"{nucleus_str}_mf_{mf_str}"