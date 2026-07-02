
from nuPhase.utils import carbon, oxygen, Molecule
from nuPhase.sample import Sample, SubSample, Parameters, Binning, NuFlavour, NuisanceFile
from nuPhase.selection import SelectionNumu0Pi1P0N, SelectionNue0Pi0P
from nuPhase.oscillator import OscillationCalculator

import typing
from argparse import ArgumentParser
import sys

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

flux_bins = np.array([
    0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,
    0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33,
    0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45,
    0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
    0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
    0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.82,
    0.84, 0.86, 0.88, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6,
    3.8, 4, 4.5, 5, 6, 7, 8, 9, 10
])

## Full mode list
# modes = {
#   "CCQE": [1],
#   "CC1pipm": [11, 3],
#   "CCcoh": [16],
#   "CCMpi": [21],
#   "CCDIS": [26],
#   "NC1pi0": [31, 32],
#   "NC1pipm": [33, 34],
#   "NCcoh": [36],
#   "NCoth": [42, 43, 44, 45, 51, 52, 35],
#   "2p2h": [2],
#   "NC1gam": [38, 39],
#   "CCMisc": [15, 17, 22, 23],
#   "NCMpi": [41],
#   "NCDIS": [46],
#   "CC1pi0": [12],
# }

modes = {
  "CCQE": [1],
  "2p2h": [2],
  "CC1pipm": [11, 3],
  "CCcoh": [16],
  "CCMpi": [21],
  "CCDIS": [26],
  "NC1pi0": [31, 32],
  "NC1pipm": [33, 34],
  "NCcoh": [36],
  "NCoth": [42, 43, 44, 45, 51, 52, 35],
  "NC1gam": [38, 39],
  "CCMisc": [15, 17, 22, 23],
  "NCMpi": [41],
  "NCDIS": [46],
  "CC1pi0": [12]
}

class PhaseSpaceAnalysis:

    def __init__(
            self, 
            out_file_name: str, 
            nd_numu: Sample, fd_nue: Sample, 
            nd_nue: Sample = None, fd_numu: Sample = None
        ):
            
        binning_check = nd_numu.binning == fd_nue.binning
        if nd_nue is not None:
            binning_check = binning_check and nd_numu.binning == nd_nue.binning
        if fd_numu is not None:
            binning_check = binning_check and nd_numu.binning == fd_nue.binning
        
        assert binning_check, "Binning must be the same for all samples!!"

        self._pdf = PdfPages(out_file_name)
        self._fig = plt.figure()

        self.nd_numu = nd_numu
        self.nd_nue = nd_nue
        self.fd_numu = fd_numu
        self.fd_nue = fd_nue

    def run(self):

        self.make_flux_plots(self.nd_numu)
        self.make_flux_plots(self.fd_nue)

        self.make_1d_rate_plots(self.nd_numu, cumulative=True, fill=True)
        self.make_1d_rate_plots(self.fd_nue, cumulative=True, fill=True)

        nd_numu_total = self.nd_numu.get_event_rates(cut = lambda event: event.mode == 1)
        fd_nue_total = self.fd_nue.get_event_rates(cut = lambda event: event.mode == 1)

        ## locations where expected n of events is at least 1
        nd_numu_total *= nd_numu_total > 1.0
        fd_nue_total *= fd_nue_total > 1.0

        nd_numu_total[nd_numu_total == 0] = np.nan
        fd_nue_total[fd_nue_total == 0] = np.nan

        fig, ax = plt.subplots()

        self.nd_numu.imshow(ax, data_override=nd_numu_total)
        self._pdf.savefig(fig)

        fig, ax = plt.subplots()

        self.fd_nue.imshow(ax, data_override=fd_nue_total)
        self._pdf.savefig(fig)

        fig, ax = plt.subplots()

        self.fd_nue.imshow(ax, data_override=self.get_unconstrained())
        ax.set_title("Unconstrained FD nue")
        self._pdf.savefig(fig)

        #fig, ax = plt.subplots()
        #ax.contour(np.meshgrid(), nd_numu, label = "ND numu")
        #ax.contour(fd_nue, label = "FD nue")
        #plt.legend()
        #self._pdf.savefig(fig)

        self._pdf.close()

    def get_unconstrained(self):

        nd_numu = self.nd_numu.get_event_rates()
        fd_nue = self.fd_nue.get_event_rates()

        fd_nue[(nd_numu >= 1) | (fd_nue < 1)] = 0

        return fd_nue

    def make_flux_plots(self, sample: Sample):

        fig = plt.figure()

        if sample.subsamples is not None:

            for subsample in sample.subsamples:

                enu = subsample.get_array("Enu_true")
                ## make basic flux plot
                plt.hist(
                    enu, 
                    bins=flux_bins, 
                    weights=np.full(
                        enu.shape, subsample.get_flux_weight() * subsample.get_pot_weight(sample.parameters.pot)), 
                        histtype="step", 
                        label=subsample.label
                    )

        else:
            pass

        
        plt.yscale("log")
        plt.legend()
        plt.xlabel("neutrino energy [GeV]")
        plt.title(f"{sample.name} Flux")
        plt.ylabel(f"Flux [1.0 / cm^2 / 50 MeV / {sample.parameters.pot:.2E} POT]")
        self._pdf.savefig(fig)

    def make_1d_rate_plots(self, sample: Sample, cumulative: bool = False, logy: bool = False, **stairs_args):

        ## plot expected event rate as fn of neutrino energy
        fig = plt.figure()

        binning = Binning(["Enu_true"], bins = [flux_bins])
        event_rate = sample.get_event_rates(binning=binning)
        plt.stairs(event_rate, flux_bins, label="total")

        event_rate[:] = 0.0
        mode_event_rates = []

        if cumulative: 
            for mode, codes in zip(list(modes.keys())[::-1], list(modes.values())[::-1]):

                event_rate += sample.get_event_rates(cut = lambda event: event.mode in codes, binning=binning)
                mode_event_rates.append(np.copy(event_rate))
            
            for mode_event_rate, mode in zip(mode_event_rates[::-1], modes.keys()):

                plt.stairs(mode_event_rate, flux_bins, label=mode, **stairs_args)

        else:
            for mode, codes in zip(list(modes.keys()), list(modes.values())):

                plt.stairs(sample.get_event_rates(cut = lambda event: event.mode in codes, binning=binning), flux_bins, label=mode, **stairs_args)

        if logy:
            plt.yscale("log")
    
        plt.legend()
        plt.xlabel("neutrino energy [GeV]")
        plt.title(f"Event rate {sample.name}")
        plt.ylabel(f"N Events / nucleon / {sample.parameters.pot:.2E} POT")
        self._pdf.savefig(fig)
        

def setup_parser():

    parser = ArgumentParser("make-plots")

    parser.add_argument(
        "--fd-nue-nue",
        type=str,
        help="FD nue -> nue filename",
        required=True
    )
    parser.add_argument(
        "--fd-numu-nue",
        type=str,
        help="numu -> nue filename",
        required=True
    )
    parser.add_argument(
        "--nd-numu",
        type=str,
        help="numu filename",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="name of output file",
        required=True
    )
    parser.add_argument(
        "--nd-mass",
        type=float,
        help="The mass of the near detector",
        required=True
    )
    parser.add_argument(
        "--fd-mass",
        type=float,
        help="The mass of the far detector",
        required=True
    )
    parser.add_argument(
        "--detector-material",
        type=str,
        help="The material of the test detector",
        choices=["oxygen", "carbon"],
        required=True
    )
    parser.add_argument(
        "--pot",
        type=float,
        help="The desired POT",
        default=1e21,
        required=False
    )

    return parser

def main():

    parser = setup_parser()

    ## parse args 
    args = parser.parse_args(sys.argv[1:])

    target_material = {"oxygen": oxygen, "carbon": carbon}[args.detector_material]
    output_file: str = args.output
    if output_file.split(".")[-1] == "pdf":
        output_file = ".".join(output_file.split(".")[:-1])

    nd_parameters = Parameters(args.pot, target_material, args.nd_mass)
    fd_parameters = Parameters(args.pot, target_material, args.fd_mass)

    ## create subsamples for each detector
    nd_numu_subsample = SubSample(
        label = "nd numu", 
        initial_flavour = NuFlavour.muon, 
        final_flavour = NuFlavour.muon, 
        target_material = target_material
    ).fill_from_file(file = NuisanceFile(args.nd_numu, pre_selection="Mode==1"), progress_bar = True)

    oscillator = OscillationCalculator(295.0, initialisation="pdg")
    
    fd_nue_nue_subsample = SubSample(
        label = "fd nue -> nue ",
        initial_flavour = NuFlavour.electron, 
        final_flavour = NuFlavour.electron, 
        target_material = target_material,
        oscillator = oscillator
    ).fill_from_file(file = NuisanceFile(args.fd_nue_nue, pre_selection="Mode==1"), progress_bar = True)

    fd_numu_nue_subsample = SubSample(
        label = "fd numu -> nue ",
        initial_flavour = NuFlavour.muon, 
        final_flavour = NuFlavour.electron, 
        target_material = target_material,
        oscillator = oscillator
    ).fill_from_file(file = NuisanceFile(args.fd_numu_nue, pre_selection="Mode==1"), progress_bar = True)
    
    binning = Binning(("q3", "q0"), (80, 80), ranges = ((0.0, 4.0), (0.0, 4.0)))

    nd_numu_sample = Sample(binning, [nd_numu_subsample], nd_parameters, name = "ND Numu")
    fd_nue_sample = Sample(binning, [fd_nue_nue_subsample, fd_numu_nue_subsample], fd_parameters, name = "FD nue")

    ## first do the analysis with no selections applied

    analysis = PhaseSpaceAnalysis(
        output_file + "-without-selections.pdf",
        nd_numu = nd_numu_sample,
        fd_nue = fd_nue_sample
    )

    analysis.run()

    analysis = PhaseSpaceAnalysis(
        output_file + "-with-selections.pdf",
        nd_numu = nd_numu_sample.apply_selection(
            SelectionNumu0Pi1P0N(muon_threshold = 0.2, pion_threshold = 0.1, proton_threshold = 0.1, neutron_threshold = 0.025),
            progress_bar = True
        ),
        fd_nue = fd_nue_sample.apply_selection(
            SelectionNue0Pi0P(electron_threshold = 0.2, pion_threshold = 0.212, proton_threshold = 1.41),
            progress_bar = True
        ),
    )

    analysis.run()

if __name__ == "__main__":
    main()
