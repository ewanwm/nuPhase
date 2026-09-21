from nuPhase.event import Event
from nuPhase.modules.base import SelectionBase

from argparse import ArgumentParser, Namespace


class SelectionCCInclusive(SelectionBase):
    """Selects events with:
    - any number of the specified lepton with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton (the one specified), proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self
    ):

        super().__init__()

    def help(self):
        return self.__doc__
    
    def _parse_args(self, args: Namespace):

        self.lepton_name = args.lepton_name
        self.lepton_pdg = args.lepton_pdg
        self.lepton_threshold = args.lepton_threshold
        self.proton_threshold = args.proton_threshold
        self.pion_threshold = args.pion_threshold
        self.neutron_threshold = args.neutron_threshold

    def _initialise(self, sample):

        self.pdg_name_map = {
            self.lepton_name: [self.lepton_pdg],
            "proton": [2212],
            "neutron": [2112],
            "pion": [211, -211],
        }

        self.thresholds = {
            self.lepton_name: self.lepton_threshold,
            "proton": self.proton_threshold,
            "neutron": self.neutron_threshold,
            "pion": self.pion_threshold,
        }

        self.name = f"nu{self.lepton_name} CC Inclusive"

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--lepton-pdg", type=int, help="The PDG code of the main lepton")
        parser.add_argument("--lepton-name", type=str, help="The name to give the main lepton. **This will affect the name of the variables that are filled by this selection**")
        parser.add_argument("--lepton-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main lepton")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        parser.add_argument("--neutron-threshold", type=float, required=False, default=0.0, help="Threshold below which any neutrons will be ignored and proton related variables will not be filled")

    def _apply(self, event: Event) -> bool:

        n_particle_map = dict(
            zip(
                [name for name in self.pdg_name_map.keys()],
                [0 for _ in range(len(self.pdg_name_map.keys()))],
            )
        )

        highest_mom_particle_map = dict(
            zip(
                [name for name in self.pdg_name_map.keys()],
                [None for _ in range(len(self.pdg_name_map.keys()))],
            )
        )

        for particle in event.particles:

            for name, pdgs in zip(self.pdg_name_map.keys(), self.pdg_name_map.values()):

                if particle.pdg in pdgs:
                    if particle.momentum > self.thresholds[name]:

                        n_particle_map[name] += 1

                        if highest_mom_particle_map[name] is None:
                            highest_mom_particle_map[name] = particle

                        elif (
                            particle.momentum > highest_mom_particle_map[name].momentum
                        ):
                            highest_mom_particle_map[name] = particle

        ## fill variables
        for name, pdgs in zip(self.pdg_name_map.keys(), self.pdg_name_map.values()):

            event.aux_vars[f"n_{name}"] = n_particle_map[name]

            if highest_mom_particle_map[name] is not None:
                event.aux_vars[f"p_{name}"] = highest_mom_particle_map[name].momentum
                event.aux_vars[f"cos_{name}"] = (
                    highest_mom_particle_map[name].three_momentum[2]
                    / highest_mom_particle_map[name].momentum
                )

            else:
                event.aux_vars[f"p_{name}"] = None
                event.aux_vars[f"cos_{name}"] = None

        ## check if there is at least one muon
        if n_particle_map[self.lepton_name] > 0:

            return True

        else:
            return False


class SelectionNumuCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of muons with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton muon, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self
    ):

        super().__init__()

        self.lepton_pdg=13
        self.lepton_name="mu"

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--muon-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main muon")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        parser.add_argument("--neutron-threshold", type=float, required=False, default=0.0, help="Threshold below which any neutrons will be ignored and proton related variables will not be filled")

    def _parse_args(self, args: Namespace):

        self.lepton_threshold=args.muon_threshold
        self.proton_threshold=args.proton_threshold
        self.pion_threshold=args.pion_threshold
        self.neutron_threshold=args.neutron_threshold
        
class SelectionNueCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of electrons with momentum > electron_threshold

    fills kinematic variables for the highest momentum electron, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self
    ):

        super().__init__()

        self.lepton_pdg=11
        self.lepton_name="e"

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--electron-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main electron")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        parser.add_argument("--neutron-threshold", type=float, required=False, default=0.0, help="Threshold below which any neutrons will be ignored and proton related variables will not be filled")

    def _parse_args(self, args: Namespace):

        self.lepton_threshold=args.electron_threshold
        self.proton_threshold=args.proton_threshold
        self.pion_threshold=args.pion_threshold
        self.neutron_threshold=args.neutron_threshold

class SelectionNumubarCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of anti-muons with momentum > muon_threshold

    fills kinematic variables for the highest momentum lepton anti-muon, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self
    ):

        super().__init__()

        self.lepton_pdg=-13
        self.lepton_name="mubar"

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--muon-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main muon")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        parser.add_argument("--neutron-threshold", type=float, required=False, default=0.0, help="Threshold below which any neutrons will be ignored and proton related variables will not be filled")

    def _parse_args(self, args: Namespace):

        self.lepton_threshold=args.muon_threshold
        self.proton_threshold=args.proton_threshold
        self.pion_threshold=args.pion_threshold
        self.neutron_threshold=args.neutron_threshold


class SelectionNuebarCCInclusive(SelectionCCInclusive):
    """Selects events with:
    - any number of positrons with momentum > electron_threshold

    fills kinematic variables for the highest momentum positron, proton and charged pion (assuming they are above the corresponding threshold)
    """

    def __init__(
        self
    ):

        super().__init__()

        self.lepton_pdg=-11
        self.lepton_name="ebar"

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--electron-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main electron")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        parser.add_argument("--neutron-threshold", type=float, required=False, default=0.0, help="Threshold below which any neutrons will be ignored and proton related variables will not be filled")

    def _parse_args(self, args: Namespace):

        self.lepton_threshold=args.electron_threshold
        self.proton_threshold=args.proton_threshold
        self.pion_threshold=args.pion_threshold
        self.neutron_threshold=args.neutron_threshold

class SelectionNue0PiNP(SelectionBase):
    """Selects events with:
    - one and only one lepton (with specified PDG) with momentum > lepton_threshold
    - N protons with momentum > proton_threshold
    - no pions (any charge) with momentum > pion_threshold
    """

    def __init__(
        self,
    ):

        super().__init__()

    def help(self):
        return self.__doc__
    
    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--lepton-pdg", type=int, required=True, help="PDG code for the main lepton")
        parser.add_argument("--lepton-name", type=str, required=True, help="The name for the main lepton. **This will change the name of the saved variables**")
        parser.add_argument("--n-protons", type=int, required=False, default=0, help="The number of protons required for the event to pass the selection")
        parser.add_argument("--lepton-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main lepton")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        
    def _parse_args(self, args: Namespace):

        self.lepton_name=args.lepton_name
        self.lepton_pdg=args.lepton_pdg
        self.lepton_threshold=args.lepton_threshold
        self.proton_threshold=args.proton_threshold
        self.pion_threshold=args.pion_threshold
        self.n_proton=args.n_protons

        self.name = f"nu{self.lepton_name} 0pi {self.n_protons} proton"

    def _apply(self, event: Event) -> bool:

        leptons_above_threshold = 0
        protons_above_threshold = 0
        pions_above_threshold = 0

        highest_mom_lepton = None

        for particle in event.particles:

            if particle.pdg == self.lepton_pdg:

                if particle.momentum > self.lepton_threshold:
                    leptons_above_threshold += 1

                    if highest_mom_lepton is None:
                        highest_mom_lepton = particle
                    elif particle.momentum > highest_mom_lepton.momentum:
                        highest_mom_lepton = particle

            elif particle.pdg == 2212:
                if particle.momentum > self.proton_threshold:
                    protons_above_threshold += 1

            elif particle.pdg in [211, -211, 111]:
                if particle.momentum > self.pion_threshold:
                    pions_above_threshold += 1

        if (
            leptons_above_threshold == 1
            and protons_above_threshold == self.n_proton
            and pions_above_threshold == 0
        ):

            event.aux_vars[f"p_{self.lepton_name}"] = highest_mom_lepton.momentum
            event.aux_vars[f"cos_{self.lepton_name}"] = (
                highest_mom_lepton.three_momentum[2] / highest_mom_lepton.momentum
            )

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
        self
    ):

        self.name = "nue 0pi 0proton"

    def help(self):
        return self.__doc__

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--electron-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main electron")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        
    def _parse_args(self, args: Namespace):

        self.electron_threshold = args.electron_threshold
        self.proton_threshold = args.proton_threshold
        self.pion_threshold = args.pion_threshold

    def _apply(self, event: Event) -> bool:

        electrons_above_threshold = 0
        protons_above_threshold = 0
        pions_above_threshold = 0

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
            electrons_above_threshold == 1
            and protons_above_threshold == 0
            and pions_above_threshold == 0
        ):

            event.aux_vars["p_e"] = highest_mom_electron.momentum
            event.aux_vars["cos_e"] = (
                highest_mom_electron.three_momentum[2] / highest_mom_electron.momentum
            )

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
        self
    ):

        self.name = "numu 0pi 0proton"

    def help(self):
        return self.__doc__

    def _setup_parser(self, parser: ArgumentParser):

        parser.add_argument("--muon-threshold", type=float, required=False, default=0.0, help="The momentum threshold of the main muon")
        parser.add_argument("--proton-threshold", type=float, required=False, default=0.0, help="Threshold below which any protons will be ignored and proton related variables will not be filled")
        parser.add_argument("--pion-threshold", type=float, required=False, default=0.0, help="Threshold below which any pions will be ignored and proton related variables will not be filled")
        
    def _parse_args(self, args: Namespace):

        self.muon_threshold = args.muon_threshold
        self.proton_threshold = args.proton_threshold
        self.pion_threshold = args.pion_threshold

    def _apply(self, event: Event) -> bool:

        muons_above_threshold = 0
        protons_above_threshold = 0
        pions_above_threshold = 0

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
            muons_above_threshold == 1
            and protons_above_threshold == 0
            and pions_above_threshold == 0
        ):

            event.aux_vars["p_mu"] = highest_mom_muon.momentum
            event.aux_vars["cos_mu"] = (
                highest_mom_muon.three_momentum[2] / highest_mom_muon.momentum
            )

            return True

        else:
            return False
