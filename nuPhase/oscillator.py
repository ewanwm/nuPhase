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

        self.parameters = {
            "theta12":None,
            "theta23":None,
            "theta13":None,
            "deltacp":None,
            "dmsq21": None,
            "dmsq32": None
        }

        self.density = density
        self.baseline = baseline

        if initialisation == "zeros":
            self.parameters["theta12"] = Tensor.zeros([1, 1]).requires_grad(True)
            self.parameters["theta23"] = Tensor.zeros([1, 1]).requires_grad(True)
            self.parameters["theta13"] = Tensor.zeros([1, 1]).requires_grad(True)
            self.parameters["deltacp"] = Tensor.zeros([1, 1]).requires_grad(True)
            self.parameters["dmsq21"]  = Tensor.zeros([1, 1]).requires_grad(True)
            self.parameters["dmsq32"]  = Tensor.zeros([1, 1]).requires_grad(True)

        elif initialisation == "pdg":
            self.parameters["theta12"] = Tensor([33.41 * m.pi / 180.0], requires_grad=True)
            self.parameters["theta23"] = Tensor([49.1  * m.pi / 180.0], requires_grad=True)
            self.parameters["theta13"] = Tensor([8.54  * m.pi / 180.0], requires_grad=True)
            self.parameters["deltacp"] = Tensor([197.0 * m.pi / 180.0], requires_grad=True)
            self.parameters["dmsq21"]  = Tensor([7.41e-5  * units.eV * units.eV], requires_grad=True)
            self.parameters["dmsq32"]  = Tensor([2.437e-3 * units.eV * units.eV], requires_grad=True)

        else:
            raise ValueError(f"Invalid initialisation option: {initialisation}")

        ## build the propagator
        self.propagator = None
        self._setup_propagator()

    def __setstate__(self, state):

        self.__dict__.update(state)

        self._setup_propagator()

    def __getstate__(self):

        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['propagator']

    def _setup_propagator(self):

        self.propagator = DPpropagator(10).set_baseline(self.baseline * units.km).set_antineutrino(False).set_density(self.density)
        self.propagator.set_theta12(self.parameters["theta12"])
        self.propagator.set_theta23(self.parameters["theta23"])
        self.propagator.set_theta13(self.parameters["theta13"])
        self.propagator.set_deltacp(self.parameters["deltacp"])
        self.propagator.set_dmsq21( self.parameters["dmsq21"] )
        self.propagator.set_dmsq31( self.parameters["dmsq32"] + self.parameters["dmsq21"] )

    def zero_grad(self):
        """Zero out the gradient of all of the parameters
        """

        for parameter in self.parameters.values():
            if parameter.grad() is not None:

                parameter.zero_grad()

    def calculate_osc_probs(self, energies: typing.Union[np.ndarray, Tensor]) -> Tensor:
        """Calculate oscillation probability for a given set of energies
        
        If energies has shape [n], returned probabilities tensor will have shape [n, 3, 3]
        """

        energies_tensor = None
        if type(energies) == np.ndarray:
            energies_tensor = Tensor(energies, dtype=dtype.scalar_type.complex_float)
        elif type(energies) == Tensor:
            energies_tensor = energies
        else:
            raise ValueError("bad type for energies, should be numpy array or nuTens Tensor")
        
        self.propagator.set_energies(energies_tensor * nt.units.GeV)
        
        return self.propagator.calculate_probs()
