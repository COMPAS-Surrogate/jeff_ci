from .mock_observation import MockObservation
from .lvk_observation import LVKObservation


def load_observation(obs_type: str, *args, **kwargs) -> 'ObservationBase':
    """
    Load an observation of the given type.

    :param obs_type: The type of observation to load. Options are 'mock' or 'lvk'.
    :param args: Positional arguments to pass to the observation constructor.
    :param kwargs: Keyword arguments to pass to the observation constructor.

    :return: An instance of the requested observation type.
    """
    obs_type = obs_type.lower()
    if obs_type == 'mock':
        return MockObservation(*args, **kwargs)
    elif obs_type == 'lvk':
        return LVKObservation(*args, **kwargs)
    else:
        raise ValueError(f"Unknown observation type: {obs_type}. Valid options are 'mock' or 'lvk'.")