from __future__ import annotations

import numpy as np

# Neijssel+19 Eq.7-9
NEIJSSEL_Z0 = 0.035
NEIJSSEL_ALPHA = -0.23
NEIJSSEL_SIGMA = 0.39
NEIJSSEL_SFR_A = 0.01
NEIJSSEL_SFR_D = 4.7
DEFAULT_PARAMS = [NEIJSSEL_ALPHA, NEIJSSEL_SIGMA, NEIJSSEL_SFR_A, NEIJSSEL_SFR_D]

# Parameter grids used by the surrogate training / bounds.
_ALPHA_ANCHORS = [-0.500, -0.400, -0.300, -0.200, -0.100, -0.001]
_SIGMA_ANCHORS = [0.100, 0.200, 0.300, 0.400, 0.500, 0.600]
_SFR_A_ANCHORS = [0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
_SFR_D_ANCHORS = [4.200, 4.400, 4.600, 4.800, 5.000, 5.200]

ALPHA_VALUES = np.linspace(min(_ALPHA_ANCHORS), max(_ALPHA_ANCHORS), 10)
SIGMA_VALUES = np.linspace(min(_SIGMA_ANCHORS), max(_SIGMA_ANCHORS), 10)
SFR_A_VALUES = np.linspace(min(_SFR_A_ANCHORS), max(_SFR_A_ANCHORS), 10)
SFR_D_VALUES = np.linspace(min(_SFR_D_ANCHORS), max(_SFR_D_ANCHORS), 10)

