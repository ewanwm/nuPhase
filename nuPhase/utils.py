import typing

import uproot

from matplotlib import pyplot as plt

import numpy as np

from oscillator import calculate_osc_probs

N_AVOGADRO = 6.02214076e23

class Molecule:

    def __init__(self, n_nucleons: int, molar_mass: float):

        self.n_nucleons = n_nucleons
        self.molar_mass = molar_mass

    def get_n_nucleons(self, mass: float = None):

        if mass is None:
            return self.n_nucleons
        
        return mass * N_AVOGADRO / self.molar_mass

carbon = Molecule(12, 12e-3)

oxygen = Molecule(16, 16e-3)
