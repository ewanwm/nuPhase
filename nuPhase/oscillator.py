import nuTens as nt
import numpy as np

from nuTens import dtype, units, tensor
from nuTens.tensor import Tensor
from nuTens.propagator import DPpropagator

import math as m
import typing

from nuPhase.event import Event

class OscillationCalculator:

    def __init__(self, baseline: float, density: float = 2.6, initialisation: str = "zeros"):

        self.theta12: Tensor = None
        self.theta23: Tensor = None
        self.theta13: Tensor = None
        self.deltacp: Tensor = None
        self.dmsq21:  Tensor = None
        self.dmsq32:  Tensor = None
        self.dmsq31:  Tensor = None

        if initialisation == "zeros":
            self.theta12 = tensor.zeros([1])
            self.theta23 = tensor.zeros([1])
            self.theta13 = tensor.zeros([1])
            self.deltacp = tensor.zeros([1])
            self.dmsq21  = tensor.zeros([1])
            self.dmsq32  = tensor.zeros([1])

        elif initialisation == "pdg":
            self.theta12 = Tensor([33.41 * m.pi / 180.0])
            self.theta23 = Tensor([49.1  * m.pi / 180.0])
            self.theta13 = Tensor([8.54  * m.pi / 180.0])
            self.deltacp = Tensor([197.0 * m.pi / 180.0])
            self.dmsq21  = Tensor([7.41e-5  * units.eV * units.eV])
            self.dmsq32  = Tensor([2.437e-3 * units.eV * units.eV])

        else:
            raise ValueError(f"Invalid initialisation option: {initialisation}")

        self.dmsq31  = self.dmsq21 + self.dmsq32

        ## build the propagator
        self.propagator = DPpropagator(10).set_baseline(baseline * units.km).set_antineutrino(False).set_density(density)
        self.propagator.set_theta12(self.theta12)
        self.propagator.set_theta23(self.theta23)
        self.propagator.set_theta13(self.theta13)
        self.propagator.set_deltacp(self.deltacp)
        self.propagator.set_dmsq21(self.dmsq21)
        self.propagator.set_dmsq31(self.dmsq31)

    def zero_grad(self):
        """Zero out the gradient of all of the parameters
        """

        self.theta12.zero_grad()
        self.theta23.zero_grad()
        self.theta13.zero_grad()
        self.deltacp.zero_grad()
        self.dmsq21.zero_grad() 
        self.dmsq32.zero_grad() 

    def calculate_osc_probs(self, energies: typing.Union[np.ndarray, Tensor]) -> Tensor:
        """Calculate oscillation probability for a given set of energies
        
        If energies has shape [n], returned probabilities tensor will have shape [n, 3, 3]
        """

        energies_tensor = None
        if type(energies) == np.ndarray:
            energies_tensor = Tensor(energies)
        elif type(energies) == Tensor:
            energies_tensor = energies
        else:
            raise ValueError("bad type for energies, should be numpy array or nuTens Tensor")
        
        self.propagator.set_energies(energies_tensor)
        
        return self.propagator.calculate_probs()
