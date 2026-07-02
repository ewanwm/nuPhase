import typing

import numpy as np

class Particle:
    """Represents a single particle within an event
    """

    def __init__(self, pdg: int, momentum: typing.Tuple[float], energy: float):

        self.pdg = pdg
        
        assert(len(momentum) == 3)
        self.three_momentum = momentum
        self.momentum = np.linalg.norm(self.three_momentum)

        self.energy = energy

    def __str__(self):
        return f"Particle(pdg = {self.pdg:<6}, mom = {self.momentum:.5f}, E = {self.energy:.5f})"

class Event:
    """represents a single event (really a single neut interaction)
    """

    def __init__(self, mode:int, e_nu: float, nu_pdg: int):

        self.n_particles: int = 0
        self.particles: typing.List[Particle] = []
        self.enu_true: float = e_nu
        self.nu_pdg_true: float = nu_pdg
        self.mode: int = mode

        ## auxilary variables that aren't necessarily needed for usual calculations
        ## but might be useful e.g. for plotting 
        self.aux_vars: typing.Dict[str, float|int] = {}

    def add_particle(self, particle: Particle) -> None:
        """Add a single particle to the event

        :param particle: particle to add
        :type particle: Particle
        """

        self.n_particles += 1

        self.particles.append(particle)

    def add_particles_from_arrays(self, pdg: np.array, energies: np.array, px: np.array, py: np.array, pz: np.array) -> None:
        """Add a number of particles to the event

        All arrays must have the same shape (which should be 1D with length == num of particles)

        :param pdg: PDG codes for each particle
        :type pdg: np.array
        :param energies: energies of each particle
        :type energies: np.array
        :param px: x momenta
        :type px: np.array
        :param py: y momenta
        :type py: np.array
        :param pz: z momenta
        :type pz: np.array
        """

        assert pdg.shape[0] == energies.shape[0] == px.shape[0] == py.shape[0] == pz.shape[0]
        assert len(pdg.shape) == len(energies.shape) == len(px.shape) == len(py.shape) == len(pz.shape) == 1

        for _pdg, _energy, _px, _py, _pz in zip(pdg, energies, px, py, pz):

            self.particles.append(Particle(pdg = _pdg, momentum = (_px, _py, _pz), energy = _energy))
            self.n_particles += 1

    def get_var(self, var: str):
        """Get the value of some event level variable

        :raises: ValueError if variable is not one of the standard ones and is not in the aux variable dictionary for this event
        """

        if var == "Enu_true":
            return self.enu_true
        elif var == "PDGnu":
            return self.nu_pdg_true
        elif var == "Mode":
            return self.mode
        elif var in self.aux_vars.keys():
            return self.aux_vars[var]
        else:
            raise ValueError(f"Variable {var} was not found in event with aux variables {self.aux_vars}")

    def __str__(self):
        return f'Event(neutrino pdg: {self.nu_pdg_true: <3}, E nu: {self.enu_true:.5f}, n particles: {len(self.particles): <2}, aux vars: {[f"{var_name}={var_value:.3f}" for var_name, var_value in zip(self.aux_vars.keys(), self.aux_vars.values())]})'