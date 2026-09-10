from nuPhase.event import Event

import abc

class SelectionBase(abc.ABC):

    def __init__(self):

        self.name = "NO NAME"

    @abc.abstractmethod
    def apply(self, event: Event) -> bool:

        raise NotImplementedError()

class SelectionNumu0PiNP0N(SelectionBase):
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
        neutron_threshold: float,
        n_protons: int = 0
    ):

        self.muon_threshold    = muon_threshold
        self.proton_threshold  = proton_threshold
        self.pion_threshold    = pion_threshold
        self.neutron_threshold = neutron_threshold

        self.name = f"numu 0pi {n_protons} proton 0 neutron"

        self.n_proton = n_protons

    def apply(self, event: Event) -> bool:

        muons_above_threshold    = 0
        protons_above_threshold  = 0
        pions_above_threshold    = 0
        neutrons_above_threshold = 0

        highest_mom_muon = None

        for particle in event.particles:

            if particle.pdg == 13:
                if particle.momentum > self.muon_threshold:
                    muons_above_threshold += 1

                    if highest_mom_muon is None:
                        highest_mom_muon = particle

                    elif particle.momentum > highest_mom_muon.momentum:
                        highest_mom_muon = particle
            
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
            muons_above_threshold    == 1 and
            protons_above_threshold  == self.n_proton and
            pions_above_threshold    == 0 and
            neutrons_above_threshold == 0
        ):

            event.aux_vars["p_mu"]   = highest_mom_muon.momentum
            event.aux_vars["cos_mu"] = highest_mom_muon.three_momentum[2] / highest_mom_muon.momentum

            return True

        else:
            return False


class SelectionCCInclusive(SelectionBase):
    """Selects events with:
    - any number of the specified lepton with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton (the one specified), proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self,
        lepton_pdg: int,
        lepton_name: str,
        lepton_threshold: float,
        proton_threshold: float,
        pion_threshold: float,
        neutron_threshold: float
    ):

        self.pdg_name_map = {
            lepton_name: [lepton_pdg],
            "proton":    [2212],
            "neutron":   [2112],  
            "pion":      [211, -211] 
        }

        self.thresholds = {
            lepton_name: lepton_threshold,
            "proton":    proton_threshold,
            "neutron":   neutron_threshold,
            "pion":      pion_threshold
        }

        self.lepton_pdg        = lepton_pdg
        self.lepton_threshold  = lepton_threshold
        self.proton_threshold  = proton_threshold
        self.pion_threshold    = pion_threshold
        self.neutron_threshold = neutron_threshold

        self.lepton_name = lepton_name

        self.name = f"nu{lepton_name} CC Inclusive"

    def apply(self, event: Event) -> bool:

        n_particle_map = dict(zip(
            [name for name in self.pdg_name_map.keys()],
            [0 for _ in range(len(self.pdg_name_map.keys()))]  
        ))

        highest_mom_particle_map = dict(zip(
            [name for name in self.pdg_name_map.keys()],
            [None for _ in range(len(self.pdg_name_map.keys()))]
        ))

        for particle in event.particles:

            for name, pdgs in zip(self.pdg_name_map.keys(), self.pdg_name_map.values()):

                if particle.pdg in pdgs:
                    if particle.momentum > self.thresholds[name]:

                        n_particle_map[name] += 1

                        if highest_mom_particle_map[name] is None:
                            highest_mom_particle_map[name] = particle

                        elif particle.momentum > highest_mom_particle_map[name].momentum:
                            highest_mom_particle_map[name] = particle

        ## fill variables
        for name, pdgs in zip(self.pdg_name_map.keys(), self.pdg_name_map.values()):

            event.aux_vars[f"n_{name}"]   = n_particle_map[name]

            if highest_mom_particle_map[name] is not None:
                event.aux_vars[f"p_{name}"]   = highest_mom_particle_map[name].momentum
                event.aux_vars[f"cos_{name}"] = highest_mom_particle_map[name].three_momentum[2] / highest_mom_particle_map[name].momentum

            else:
                event.aux_vars[f"p_{name}"]   = None
                event.aux_vars[f"cos_{name}"] = None

        ## check if there is at least one muon                
        if (
            n_particle_map[self.lepton_name] > 0
        ):

            return True

        else:
            return False

class SelectionNumuCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of muons with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton muon, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self,
        muon_threshold: float,
        proton_threshold: float,
        pion_threshold: float,
        neutron_threshold: float
    ):

        super().__init__(
            lepton_pdg = 13, 
            lepton_name = "mu", 
            lepton_threshold = muon_threshold, 
            proton_threshold = proton_threshold, 
            pion_threshold = pion_threshold, 
            neutron_threshold = neutron_threshold
        )

class SelectionNueCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of electrons with momentum > electron_threshold

    fills kinematic variables for the highest momentum electron, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self,
        electron_threshold: float,
        proton_threshold: float,
        pion_threshold: float,
        neutron_threshold: float
    ):

        super().__init__(
            lepton_pdg = 11, 
            lepton_name = "e", 
            lepton_threshold = electron_threshold, 
            proton_threshold = proton_threshold, 
            pion_threshold = pion_threshold, 
            neutron_threshold = neutron_threshold
        )

class SelectionNumubarCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of anti-muons with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton anti-muon, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self,
        muon_threshold: float,
        proton_threshold: float,
        pion_threshold: float,
        neutron_threshold: float
    ):

        super().__init__(
            lepton_pdg = -13, 
            lepton_name = "mubar", 
            lepton_threshold = muon_threshold, 
            proton_threshold = proton_threshold, 
            pion_threshold = pion_threshold, 
            neutron_threshold = neutron_threshold
        )

class SelectionNuebarCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of positrons with momentum > electron_threshold

    fills kinematic variables for the highest momentum positron, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self,
        electron_threshold: float,
        proton_threshold: float,
        pion_threshold: float,
        neutron_threshold: float
    ):

        super().__init__(
            lepton_pdg = -11, 
            lepton_name = "ebar", 
            lepton_threshold = electron_threshold, 
            proton_threshold = proton_threshold, 
            pion_threshold = pion_threshold, 
            neutron_threshold = neutron_threshold
        )

class SelectionNue0PiNP0N(SelectionBase):
    """Selects events with:
    - one and only one electron with momentum > electron_threshold
    - one and only one proton with momentum > proton_threshold
    - no pions (any charge) with momentum > pion_threshold
    - no neutrons with momentum > neutron_threshold
    """

    def __init__(
        self, 
        electron_threshold: float, 
        proton_threshold: float, 
        pion_threshold: float, 
        neutron_threshold: float,
        n_protons: int = 0
    ):

        self.electron_threshold = electron_threshold
        self.proton_threshold   = proton_threshold
        self.pion_threshold     = pion_threshold
        self.neutron_threshold  = neutron_threshold

        self.name = f"nue 0pi {n_protons} proton 0 neutron"

        self.n_proton = n_protons

    def apply(self, event: Event) -> bool:

        electrons_above_threshold = 0
        protons_above_threshold   = 0
        pions_above_threshold     = 0
        neutrons_above_threshold  = 0

        highest_mom_electron = None

        for particle in event.particles:

            if particle.pdg == 11:

                if particle.momentum > self.electron_threshold:
                    electrons_above_threshold += 1

                    if highest_mom_electron is None:
                        highest_mom_electron = particle
                    elif particle.momentum > highest_mom_electron.momentum:
                        highest_mom_electron = particle
                
            
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
            electrons_above_threshold == 1 and
            protons_above_threshold   == self.n_proton and
            pions_above_threshold     == 0 and
            neutrons_above_threshold  == 0
        ):

            event.aux_vars["p_e"]   = highest_mom_electron.momentum
            event.aux_vars["cos_e"] = highest_mom_electron.three_momentum[2] / highest_mom_electron.momentum

            return True

        else:
            return False

class SelectionNue0Pi0P(SelectionBase):
    """Selects events with:
    - one and only one electron or positron with momentum > electron_threshold
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

        highest_mom_electron = None

        for particle in event.particles:

            if abs(particle.pdg) == 11:
                if particle.momentum > self.electron_threshold:
                    electrons_above_threshold += 1

                if highest_mom_electron is None:
                    highest_mom_electron = particle
                elif particle.momentum > highest_mom_electron.momentum:
                    highest_mom_electron = particle
            
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

            event.aux_vars["p_e"]   = highest_mom_electron.momentum
            event.aux_vars["cos_e"] = highest_mom_electron.three_momentum[2] / highest_mom_electron.momentum
            
            return True

        else:
            return False

class SelectionNumu0Pi0P(SelectionBase):
    """Selects events with:
    - one and only one muon or anti-muon with momentum > muon_threshold
    - no protons with momentum > proton_threshold
    - no pions (any charge) with momentum > pion_threshold
    """

    def __init__(
        self, 
        muon_threshold: float, 
        proton_threshold: float, 
        pion_threshold: float
    ):

        self.muon_threshold = muon_threshold
        self.proton_threshold  = proton_threshold
        self.pion_threshold    = pion_threshold

        self.name = "numu 0pi 0proton"

    def apply(self, event: Event) -> bool:

        muons_above_threshold     = 0
        protons_above_threshold   = 0
        pions_above_threshold     = 0

        highest_mom_muon = None

        for particle in event.particles:

            if abs(particle.pdg) == 13:
                if particle.momentum > self.muon_threshold:
                    muons_above_threshold += 1
                    
                    if highest_mom_muon is None:
                        highest_mom_muon = particle

                    elif particle.momentum > highest_mom_muon.momentum:
                        highest_mom_muon = particle
            
            elif particle.pdg == 2212:
                if particle.momentum > self.proton_threshold:
                    protons_above_threshold += 1
                    
            elif particle.pdg in [211, -211, 111]:
                if particle.momentum > self.pion_threshold:
                    pions_above_threshold += 1

        if (
            muons_above_threshold     == 1 and
            protons_above_threshold   == 0 and
            pions_above_threshold     == 0
        ):

            event.aux_vars["p_mu"]   = highest_mom_muon.momentum
            event.aux_vars["cos_mu"] = highest_mom_muon.three_momentum[2] / highest_mom_muon.momentum

            return True

        else:
            return False
