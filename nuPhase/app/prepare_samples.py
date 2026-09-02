
from nuPhase.utils import strip_file_extension
from nuPhase.materials import carbon, oxygen, water
from nuPhase.sample import Sample, SubSample, Parameters, Binning, NuFlavour, NuisanceFile
from nuPhase.modules.selection import SelectionNumu0Pi1P0N, SelectionNue0Pi0P, SelectionNumu0Pi0P, SelectionNue0Pi1P0N
from nuPhase.oscillator import OscillationCalculator
from nuPhase.analysis import FisherInfoAnalysis, BasicAnalysis, UnconstrainableNueAnalysis
from nuPhase.modules.transformations import CalculateFisherInfo

import typing
from argparse import ArgumentParser
import sys

import numpy as np

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
        "--fd-numu-numu",
        type=str,
        help="numu -> numu filename",
        required=False
    )
    parser.add_argument(
        "--nd-numu",
        type=str,
        help="numu filename",
        required=True
    )
    parser.add_argument(
        "--nd-nue",
        type=str,
        help="numu filename",
        required=False
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
        choices=["oxygen", "carbon", "water"],
        required=True
    )
    parser.add_argument(
        "--pot",
        type=float,
        help="The desired POT",
        default=1e21,
        required=False
    )
    parser.add_argument(
        "-n", "--max_events_per_file",
        type=int,
        help="A maximum number of events to read in from each input file",
        default=np.inf,
        required=False
    )
    parser.add_argument(
        "--fisher-info-analysis",
        action="store_true",
        help="Do fisher information analysis on samples with selections applied",
        default=False
    )

    return parser

def main():

    parser = setup_parser()

    ## parse args 
    args = parser.parse_args(sys.argv[1:])

    target_material = {"oxygen": oxygen, "carbon": carbon, "water": water}[args.detector_material]
    output_file: str = strip_file_extension(args.output, "pdf")

    nd_parameters = Parameters(args.pot, target_material, args.nd_mass)
    fd_parameters = Parameters(args.pot, target_material, args.fd_mass)

    ## create subsamples for each detector
    nd_numu_subsample = SubSample(
        label = "nd numu", 
        initial_flavour = NuFlavour.muon, 
        final_flavour = NuFlavour.muon, 
        target_material = target_material
    ).fill_from_file(file = NuisanceFile(args.nd_numu), progress_bar = True, max_n_events = args.max_events_per_file)

    nd_nue_subsample = SubSample(
        label = "nd nue", 
        initial_flavour = NuFlavour.electron, 
        final_flavour = NuFlavour.electron,
        target_material = target_material
    ).fill_from_file(file = NuisanceFile(args.nd_nue), progress_bar = True, max_n_events = args.max_events_per_file)

    oscillator = OscillationCalculator(295.0, initialisation="pdg")
    
    fd_nue_nue_subsample = SubSample(
        label = "fd nue -> nue ",
        initial_flavour = NuFlavour.electron, 
        final_flavour = NuFlavour.electron, 
        target_material = target_material,
        oscillator = oscillator
    ).fill_from_file(file = NuisanceFile(args.fd_nue_nue), progress_bar = True, max_n_events = args.max_events_per_file)

    fd_numu_nue_subsample = SubSample(
        label = "fd numu -> nue ",
        initial_flavour = NuFlavour.muon, 
        final_flavour = NuFlavour.electron, 
        target_material = target_material,
        oscillator = oscillator
    ).fill_from_file(file = NuisanceFile(args.fd_numu_nue), progress_bar = True, max_n_events = args.max_events_per_file)

    fd_numu_numu_subsample = SubSample(
        label = "fd numu -> numu ",
        initial_flavour = NuFlavour.muon, 
        final_flavour = NuFlavour.muon, 
        target_material = target_material,
        oscillator = oscillator
    ).fill_from_file(file = NuisanceFile(args.fd_numu_numu), progress_bar = True, max_n_events = args.max_events_per_file)
    
    nd_nue_binning  = Binning(("p_e", "cos_e"), (50, 50), ranges = ((0.0, 1.0), (-1.0, 1.0)))
    nd_numu_binning = Binning(("p_mu", "cos_mu"), (50, 50), ranges = ((0.0, 1.0), (-1.0, 1.0)))

    fd_enu_edges = np.array([*np.linspace(0.2, 1.2, 10), 2.0])
    fd_cos_edges = np.array([-1.0, *np.linspace(0.0, 1.0, 10)])

    fd_nue_binning  = Binning(("Enu_true", "cos_e"),  bins = [fd_enu_edges, fd_cos_edges])
    fd_numu_binning = Binning(("Enu_true", "cos_mu"), bins = [fd_enu_edges, fd_cos_edges])

    nd_numu_sample = Sample(nd_numu_binning, [nd_numu_subsample], nd_parameters, name = "ND Numu")
    nd_nue_sample  = Sample(nd_nue_binning,  [nd_nue_subsample], nd_parameters, name = "ND Nue")
    fd_nue_sample  = Sample(fd_nue_binning,  [fd_nue_nue_subsample, fd_numu_nue_subsample], fd_parameters, name = "FD nue")
    fd_numu_sample = Sample(fd_numu_binning, [fd_numu_numu_subsample], fd_parameters, name = "FD numu")

    ## apply selections to the samples
    selected_nd_numu_sample = nd_numu_sample.apply_selection(
        SelectionNumu0Pi1P0N(muon_threshold = 0.2, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025),
        progress_bar = True
    )
    selected_nd_nue_sample = nd_nue_sample.apply_selection(
        SelectionNue0Pi1P0N(electron_threshold = 0.2, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025),
        progress_bar = True
    )
    selected_fd_nue_sample = fd_nue_sample.apply_selection(
        SelectionNue0Pi0P(electron_threshold = 0.2, pion_threshold = 0.212, proton_threshold = 1.41),
        progress_bar = True
    )
    selected_fd_numu_sample = fd_numu_sample.apply_selection(
        SelectionNumu0Pi0P(muon_threshold = 0.2, pion_threshold = 0.212, proton_threshold = 1.41),
        progress_bar = True
    )

    fisher_info_calculator = CalculateFisherInfo(oscillator = oscillator, make_plots = True, plot_file_name = strip_file_extension(output_file, "pdf") + "-fisher-info.pdf")

    fisher_info_calculator.apply(selected_fd_nue_sample)
    fisher_info_calculator.apply(selected_fd_numu_sample)

    fisher_info_calculator.finalise()
    
    selected_nd_numu_sample.to_file("nd_numu.nps")
    selected_nd_nue_sample.to_file("nd_nue.nps")
    selected_fd_numu_sample.to_file("fd_numu.nps")
    selected_fd_nue_sample.to_file("fd_nue.nps")
    
if __name__ == "__main__":
    main()
