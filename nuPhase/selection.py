from event import Event, Particle

import abc

class SelectionBase(abc.ABC):

    def __init__(self):

        self.name = "NO NAME"

    @abc.abstractmethod
    def apply(self, event: Event) -> bool:

        raise NotImplementedError()

class SelectionNumu0Pi1P0N(SelectionBase):
    """Selects events with:
    - one and only one muon with momentum > muon_threshold
    - one and only one proton with momentum > proton_threshold
    - no pions (any charge) with momentum > pion_threshold
    - no neutrons with momentum > neutron_threshold
    """

    def __init__(
        self, 
        muon_threshold: float, 
        proton_threshold: float, 
        pion_threshold: float, 
        neutron_threshold: float
    ):

        self.muon_threshold    = muon_threshold
        self.proton_threshold  = proton_threshold
        self.pion_threshold    = pion_threshold
        self.neutron_threshold = neutron_threshold

        self.name = "numu 0pi 1proton 0 neutron"

    def apply(self, event: Event) -> bool:

        muons_above_threshold    = 0
        protons_above_threshold  = 0
        pions_above_threshold    = 0
        neutrons_above_threshold = 0

        for particle in event.particles:

            if particle.pdg == 13:
                if particle.momentum > self.muon_threshold:
                    muons_above_threshold += 1
            
            elif particle.pdg == 2212:
                if particle.momentum > self.proton_threshold:
                    protons_above_threshold += 1
                    
            elif particle.pdg == 2112:
                if particle.momentum > self.neutron_threshold:
                    neutrons_above_threshold += 1

            elif particle.pdg in [211, -211, 111]:
                if particle.momentum > self.pion_threshold:
                    pions_above_threshold += 1

        if (
            muons_above_threshold     == 1 and
            protons_above_threshold  == 1 and
            pions_above_threshold    == 0 and
            neutrons_above_threshold == 0
        ):
            return True

        else:
            return False


class SelectionNue0Pi0P(SelectionBase):
    """Selects events with:
    - one and only one electron with momentum > muon_threshold
    - no protons with momentum > proton_threshold
    - no pions (any charge) with momentum > pion_threshold
    """

    def __init__(
        self, 
        electron_threshold: float, 
        proton_threshold: float, 
        pion_threshold: float
    ):

        self.electron_threshold = electron_threshold
        self.proton_threshold  = proton_threshold
        self.pion_threshold    = pion_threshold

        self.name = "nue 0pi 0proton"

    def apply(self, event: Event) -> bool:

        electrons_above_threshold = 0
        protons_above_threshold   = 0
        pions_above_threshold     = 0

        for particle in event.particles:

            if particle.pdg == 11:
                if particle.momentum > self.electron_threshold:
                    electrons_above_threshold += 1
            
            elif particle.pdg == 2212:
                if particle.momentum > self.proton_threshold:
                    protons_above_threshold += 1
                    
            elif particle.pdg in [211, -211, 111]:
                if particle.momentum > self.pion_threshold:
                    pions_above_threshold += 1

        if (
            electrons_above_threshold == 1 and
            protons_above_threshold   == 0 and
            pions_above_threshold     == 0
        ):
            return True

        else:
            return False
