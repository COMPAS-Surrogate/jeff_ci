from .mock_observation import MockObservation
from .lvk_observation import LVKObservation
from .observation_base import ObservationBase


def load_observation(fname:str) -> ObservationBase:
    obs_type = 'mock' if 'mock' in fname else 'lvk' if 'lvk' in fname.lower() else None
    if obs_type == 'mock':
        return MockObservation.load_h5(fname)
    elif obs_type == 'lvk':
        return LVKObservation.load_h5(fname)
    else:
        raise ValueError(f"Unknown observation type in filename: {fname}. Must contain 'mock' or 'lvk'.")