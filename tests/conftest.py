import os
import shutil
import urllib.request
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

np.random.seed(0)

HERE = Path(__file__).resolve().parent
TEST_DATA = HERE / "test_data"
LARGE_TEST_DATA = HERE / "large_test_data"
TEST_DATA.mkdir(parents=True, exist_ok=True)

OUT_TEST = HERE / "out"
OUT_TEST.mkdir(parents=True, exist_ok=True)



@pytest.fixture
def test_compas_h5():
    """
    Fixture to provide the path to the COMPAS test data file.
    """

    external = os.environ.get("COSMIC_INTEGRATION_COMPAS_H5")
    if external:
        external_path = Path(external).expanduser()
        if external_path.exists():
            return str(external_path)

    large_test_fn = LARGE_TEST_DATA / "h5out_5M.h5"
    if large_test_fn.exists():
        return str(large_test_fn)

    path = TEST_DATA / "test_compas.h5"
    if not path.exists():
        _generate_fake_compas_file(str(path))
    return str(path)

@pytest.fixture
def outdir():
    return OUT_TEST


@pytest.fixture
def mock_sys_argv():
    """
    Pytest fixture to temporarily replace sys.argv for a test.
    Usage:
        def test_example(mock_sys_argv):
            mock_sys_argv(['my_script.py', '--option', 'value'])
            ...
    """

    def _mock_argv(new_args):
        print(f"Mocking sys.argv with: << {' '.join(new_args)} >>", )
        return patch('sys.argv', new_args)

    return _mock_argv


@pytest.fixture
def observation_file(outdir):
    mock_obs = Path(outdir) / "mock_observation.h5"
    if not mock_obs.exists():
        _download_file(
            url = 'https://github.com/COMPAS-Surrogate/ilya_simulation/raw/refs/heads/main/mock_population_weights.h5',
            dest = str(mock_obs)
        )
    return str(mock_obs)


def _generate_fake_compas_file(filename: str, n_systems=5000, frac_bbh: float = 0.7, frac_bns: float = 0.2,
                               frac_bhns: float = 0.1, ):
    m1 = np.random.uniform(3, 150, size=n_systems)
    m2 = np.random.uniform(0.1, 100, size=n_systems)

    # draw binary masses
    n_dcos = n_systems // 2
    n_ce = n_systems * 2
    types = np.random.choice(["BBH", "BNS", "NSBH"], size=n_dcos,
                             p=[frac_bbh, frac_bns, frac_bhns])

    # Define the type-to-mass mapping
    type_to_pair = {
        "BBH": [14, 14],
        "BNS": [13, 13],
        "NSBH": [13, 14]
    }

    # Create a 2D array by mapping each type to its corresponding mass pair
    mass_pairs = np.array([type_to_pair[t] for t in types]).T

    # create file structure
    with h5py.File(filename, "w") as f:
        f.create_group("BSE_System_Parameters")
        f.create_group("BSE_Common_Envelopes")
        f.create_group("BSE_Double_Compact_Objects")
        seeds = np.arange(n_systems)
        f["BSE_System_Parameters"].create_dataset("SEED", data=seeds)
        f["BSE_System_Parameters"].create_dataset("Metallicity@ZAMS(1)", data=np.random.uniform(1e-4, 1e-2, n_systems))
        f["BSE_System_Parameters"].create_dataset("Mass@ZAMS(1)", data=m1)
        f["BSE_System_Parameters"].create_dataset("Mass@ZAMS(2)", data=m2)
        f["BSE_System_Parameters"].create_dataset("Stellar_Type@ZAMS(1)", data=[16] * n_systems)
        f["BSE_System_Parameters"].create_dataset("Stellar_Type@ZAMS(2)", data=[16] * n_systems)
        f['BSE_System_Parameters'].create_dataset("CH_on_MS(1)", data=np.ones(n_systems, dtype=bool))
        f['BSE_System_Parameters'].create_dataset("CH_on_MS(2)", data=np.ones(n_systems, dtype=bool))

        # CE
        ce_seeds = np.arange(n_ce)
        f["BSE_Common_Envelopes"].create_dataset("SEED", data=ce_seeds)
        f["BSE_Common_Envelopes"].create_dataset("Immediate_RLOF>CE",
                                                 data=np.zeros(n_ce, dtype=bool))  # no RLOF after CE
        f["BSE_Common_Envelopes"].create_dataset("Optimistic_CE", data=np.zeros(n_ce, dtype=bool))  # no optimistic CE
        # DCOs
        dco_seeds = np.arange(n_dcos)
        f["BSE_Double_Compact_Objects"].create_dataset("Stellar_Type(1)", data=mass_pairs[0, :])
        f["BSE_Double_Compact_Objects"].create_dataset("Stellar_Type(2)", data=mass_pairs[1, :])
        f["BSE_Double_Compact_Objects"].create_dataset("SEED", data=dco_seeds)
        f["BSE_Double_Compact_Objects"].create_dataset("Mass(1)", data=m1[:n_dcos])
        f["BSE_Double_Compact_Objects"].create_dataset("Mass(2)", data=m2[:n_dcos])
        f["BSE_Double_Compact_Objects"].create_dataset("Time", data=np.random.uniform(4, 13.8, n_dcos))
        f["BSE_Double_Compact_Objects"].create_dataset("Coalescence_Time", data=np.random.uniform(0, 14000, n_dcos))
        f["BSE_Double_Compact_Objects"].create_dataset("Merges_Hubble_Time", data=np.ones(n_dcos, dtype=bool))


def _download_file(url: str, dest: str):
    """
    Download a file from a URL to a specified destination.
    """
    with urllib.request.urlopen(url) as response:  # noqa: S310
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)
    print(f"Downloaded {url} to {dest}")
