import os
from unittest.mock import patch

import h5py
import numpy as np
import pytest

np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = os.path.join(HERE, "test_data")
OUTDIR = os.path.join(HERE, "out")
os.makedirs(TEST_DATA, exist_ok=True)


@pytest.fixture
def test_compas_h5():
    """
    Fixture to provide the path to the COMPAS test data file.
    """
    path = os.path.join(TEST_DATA, "test_compas.h5")
    if not os.path.exists(path):
        _generate_fake_compas_file(path)
    return path


@pytest.fixture
def outdir():
    """
    Fixture to provide the output directory for tests.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    return OUTDIR


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
def observation_file(test_compas_h5, outdir, mock_sys_argv):
    """
    Fixture to ensure the rate file is generated before running tests.
    """

    rate_file_path = f"{outdir}/observation"
    if not os.path.exists(rate_file_path + '.csv'):
        from cosmic_integration import ratesSampler

        param_alpha = np.mean(ratesSampler.ALPHA_VALUES)
        param_sigma = np.mean(ratesSampler.SIGMA_VALUES)
        param_sfra = np.mean(ratesSampler.SFR_A_VALUES)
        param_sfrd = np.mean(ratesSampler.SFR_D_VALUES)

        command = (
            "python_ratesSampler.py "
            f"-i {os.path.basename(test_compas_h5)} "
            f"-p {os.path.dirname(test_compas_h5)} "
            f"-a {param_alpha} "
            f"-s {param_sigma} "
            f"-A {param_sfra} "
            f"-D {param_sfrd} "
            "-n 1 "
            f"{rate_file_path}"
        )

        with mock_sys_argv(command.split()):
            ratesSampler.main()
    return rate_file_path + '.csv'


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
