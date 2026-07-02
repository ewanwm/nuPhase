import nuTens as nt
import numpy as np

from nuTens import dtype, units, tensor
from nuTens.tensor import Tensor
from nuTens.propagator import DPpropagator

import math as m

def calculate_osc_probs(numpy_energies: np.array, baseline):
    """ calculate oscillation probabilities
    TODO make this more of a thing
    
    """
    energies = Tensor(numpy_energies, dtype.scalar_type.float, dtype.device_type.cpu) * units.GeV

    ## PDG values
    theta12 = Tensor([33.41 * m.pi / 180.0])
    theta23 = Tensor([49.1  * m.pi / 180.0])
    theta13 = Tensor([8.54  * m.pi / 180.0])
    deltacp = Tensor([197.0 * m.pi / 180.0])
    dmsq21  = Tensor([7.41e-5  * units.eV * units.eV])
    dmsq32  = Tensor([2.437e-3 * units.eV * units.eV])
    dmsq31  = dmsq21 + dmsq32

    ## build the propagator
    propagator = DPpropagator(10).set_baseline(baseline * units.km).set_antineutrino(False).set_density(2.6)
    propagator.set_theta12(theta12).set_theta23(theta23).set_theta13(theta13).set_deltacp(deltacp).set_dmsq21(dmsq21).set_dmsq31(dmsq31)

    propagator.set_energies(energies)

    return propagator.calculate_probs().numpy()