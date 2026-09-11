
from nuPhase.utils import strip_file_extension
from nuPhase.materials import carbon, oxygen, water
from nuPhase.sample import Sample, SubSample, Parameters, Binning, NuFlavour, NuisanceFile
from nuPhase.modules.selection import SelectionNumu0Pi1P0N, SelectionNue0Pi0P, SelectionNumu0Pi0P, SelectionNue0Pi1P0N
from nuPhase.oscillator import OscillationCalculator
from nuPhase.analysis import FisherInfoAnalysis, BasicAnalysis, UnconstrainableNueAnalysis

import typing
from argparse import ArgumentParser
import sys

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

def setup_parser():

    parser = ArgumentParser("make-plots")

    parser.add_argument(
        "--fd-nue",
        type=str,
        help="FD nue sample",
        required=True
    )
    parser.add_argument(
        "--fd-numu",
        type=str,
        help="FD numu sample",
        required=True
    )
    parser.add_argument(
        "--nd-numu",
        type=str,
        help="ND numu sample",
        required=True
    )
    parser.add_argument(
        "--nd-nue",
        type=str,
        help="ND nue sample",
        required=False
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="name of output file",
        required=True
    )

    return parser

def main():

    parser = setup_parser()

    ## parse args 
    args = parser.parse_args(sys.argv[1:])

    output_file: str = strip_file_extension(args.output, "pdf")

    nd_numu_sample = Sample.from_file(args.nd_numu)
    nd_nue_sample  = Sample.from_file(args.nd_nue)

    fd_nue_sample  = Sample.from_file(args.fd_nue)
    fd_numu_sample = Sample.from_file(args.fd_numu)

    oscillator = OscillationCalculator(295.0, initialisation="pdg")

    for subsample in fd_nue_sample.subsamples:
        subsample.oscillator = oscillator
    for subsample in fd_numu_sample.subsamples:
            subsample.oscillator = oscillator

    BasicAnalysis(
        output_file + "-basic-plots-with-selections.pdf",
        samples = [
            nd_numu_sample,
            nd_nue_sample,
            fd_nue_sample,
            fd_numu_sample
        ]
    ).run()

    UnconstrainableNueAnalysis(
        output_file + "-unconstrainable-nue.pdf",
        nd_numu = nd_numu_sample,
        nd_nue  = nd_nue_sample,
        fd_nue  = fd_nue_sample,
        interaction_space = Binning(["q3", "q0"], bins = [np.linspace(0, 2.0, 50), np.linspace(0, 2.0, 50)])
    ).run()
    
    analysis = FisherInfoAnalysis(
        output_file + "-fisher-analysis-with-selections.pdf",
        nd_numu = nd_numu_sample,
        nd_nue  = nd_nue_sample,
        fd_nue  = fd_nue_sample,
        fd_numu = fd_numu_sample,
        interaction_space = Binning(["Enu_true", "q3", "q0"], bins = [flux_bins, np.linspace(0, 2.0, 50), np.linspace(0, 2.0, 50)]),
        oscillator = oscillator
    )

    analysis.plot_fisher_info_map(variables=["q3", "q0"], slice_var="Enu_true")
    analysis.plot_fisher_info_map(variables=["q3", "q0"], slice_var="Enu_true", avg_per_event=True)
    analysis.run()

if __name__ == "__main__":
    main()
