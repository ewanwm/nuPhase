from nuPhase.utils import strip_file_extension
from nuPhase.materials import carbon, oxygen, water
from nuPhase.sample import (
    Sample,
    SubSample,
    Parameters,
    Binning,
    NuFlavour,
    NuisanceFile,
)
from nuPhase.modules.selection import (
    SelectionNue0Pi0P,
    SelectionNumu0Pi0P,
    SelectionNueCCInclusive,
    SelectionNumuCCInclusive,
    SelectionNuebarCCInclusive,
    SelectionNumubarCCInclusive,
)
from nuPhase.oscillator import OscillationCalculator
from nuPhase.modules.transformations import CalculateFisherInfo, ApplyVariableSmearing

import typing
from argparse import ArgumentParser
import sys

import numpy as np


def setup_parser():

    parser = ArgumentParser("make-plots")

    parser.add_argument(
        "--fd-nue-nue", type=str, help="FD nue -> nue filename", required=False
    )
    parser.add_argument(
        "--fd-numu-nue", type=str, help="numu -> nue filename", required=False
    )
    parser.add_argument(
        "--fd-numu-numu", type=str, help="numu -> numu filename", required=False
    )
    parser.add_argument("--nd-numu", type=str, help="numu filename", required=False)
    parser.add_argument("--nd-nue", type=str, help="numu filename", required=False)
    parser.add_argument(
        "-o", "--output", type=str, help="name of output file", required=True
    )
    parser.add_argument(
        "--nd-mass", type=float, help="The mass of the near detector", required=True
    )
    parser.add_argument(
        "--fd-mass", type=float, help="The mass of the far detector", required=True
    )
    parser.add_argument(
        "--detector-material",
        type=str,
        help="The material of the test detector",
        choices=["oxygen", "carbon", "water"],
        required=True,
    )
    parser.add_argument(
        "--pot", type=float, help="The desired POT", default=1e21, required=False
    )
    parser.add_argument(
        "-n",
        "--max_events_per_file",
        type=int,
        help="A maximum number of events to read in from each input file",
        default=np.inf,
        required=False,
    )
    parser.add_argument(
        "--smear-energies",
        action="store_true",
        help="Apply true -> reco smearing on neutrino energies",
        default=False,
    )
    parser.add_argument(
        "--qe-energies",
        action="store_true",
        help="use the QE 'reconstructed' energies for FD samples",
        default=False,
    )
    parser.add_argument(
        "--antinu",
        action="store_true",
        help="interpret the samples as antineutrino samples",
        default=False,
    )

    return parser


def main():

    parser = setup_parser()

    ## parse args
    args = parser.parse_args(sys.argv[1:])

    target_material = {"oxygen": oxygen, "carbon": carbon, "water": water}[
        args.detector_material
    ]
    output_file: str = strip_file_extension(args.output, "pdf")

    nd_parameters = Parameters(args.pot, target_material, args.nd_mass)
    fd_parameters = Parameters(args.pot, target_material, args.fd_mass)

    fd_enu_edges = np.array([*np.linspace(0.2, 1.2, 10), 2.0])
    fd_cos_edges = np.array([-1.0, *np.linspace(0.0, 1.0, 10)])

    fd_nue_binning = Binning(
        ("Enu_true",),
        bins=[
            fd_enu_edges,
        ],
    )
    fd_numu_binning = Binning(
        ("Enu_true",),
        bins=[
            fd_enu_edges,
        ],
    )

    if args.smear_energies:
        fd_nue_binning = Binning(
            ("Enu_reco",),
            bins=[
                fd_enu_edges,
            ],
        )
        fd_numu_binning = Binning(
            ("Enu_reco",),
            bins=[
                fd_enu_edges,
            ],
        )

    if args.qe_energies:
        fd_nue_binning = Binning(
            ("Enu_QE",),
            bins=[
                fd_enu_edges,
            ],
        )
        fd_numu_binning = Binning(
            ("Enu_QE",),
            bins=[
                fd_enu_edges,
            ],
        )

    ## stuff for calculating fisher info
    oscillator = OscillationCalculator(295.0, initialisation="pdg")
    fisher_info_calculator = CalculateFisherInfo(
        oscillator=oscillator,
        make_plots=True,
        plot_file_name=strip_file_extension(output_file, "pdf") + "-fisher-info.pdf",
    )

    ## applies a 20% energy resolution
    fd_energy_smear = ApplyVariableSmearing("Enu_true", "Enu_reco", lambda x: 0.2 * x)

    ## define selections
    if args.antinu:
        nd_numu_selection = SelectionNumubarCCInclusive(
            0.0, 0.0, 0.0, 0.0
        )  # muon_threshold = 0.1, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025)
        nd_nue_selection = SelectionNuebarCCInclusive(
            0.0, 0.0, 0.0, 0.0
        )  # electron_threshold = 0.1, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025)

        nd_nue_binning = Binning(
            ("p_ebar", "cos_ebar"), (50, 50), ranges=((0.0, 1.0), (-1.0, 1.0))
        )
        nd_numu_binning = Binning(
            ("p_mubar", "cos_mubar"), (50, 50), ranges=((0.0, 1.0), (-1.0, 1.0))
        )

    else:
        nd_numu_selection = SelectionNumuCCInclusive(
            0.0, 0.0, 0.0, 0.0
        )  # muon_threshold = 0.1, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025)
        nd_nue_selection = SelectionNueCCInclusive(
            0.0, 0.0, 0.0, 0.0
        )  # (electron_threshold = 0.1, pion_threshold = 0.1, proton_threshold = 0.35, neutron_threshold = 0.025)

        nd_nue_binning = Binning(
            ("p_e", "cos_e"), (50, 50), ranges=((0.0, 1.0), (-1.0, 1.0))
        )
        nd_numu_binning = Binning(
            ("p_mu", "cos_mu"), (50, 50), ranges=((0.0, 1.0), (-1.0, 1.0))
        )

    fd_numu_selection = SelectionNumu0Pi0P(
        muon_threshold=0.2, pion_threshold=0.212, proton_threshold=1.41
    )
    fd_nue_selection = SelectionNue0Pi0P(
        electron_threshold=0.1, pion_threshold=0.212, proton_threshold=1.41
    )

    if args.nd_numu is not None:
        nd_numu_sample = Sample(
            nd_numu_binning,
            [
                ## ND numu subsample
                SubSample(
                    label="nd numubar" if args.antinu else "nd numu",
                    initial_flavour=NuFlavour.muon,
                    final_flavour=NuFlavour.muon,
                    target_material=target_material,
                    antineutrino=args.antinu,
                ).fill_from_file(
                    file=NuisanceFile(args.nd_numu),
                    progress_bar=True,
                    max_n_events=args.max_events_per_file,
                )
            ],
            nd_parameters,
            name="ND numubar" if args.antinu else "ND numu",
            ## now apply selection to this sample
        ).apply_selection(nd_numu_selection, progress_bar=True)

        print(
            f"events in sample {nd_numu_sample.name}: {np.sum(nd_numu_sample.get_event_rates())}"
        )

        nd_numu_sample.to_file("nd_numubar.nps" if args.antinu else "nd_numu.nps")

    if args.nd_nue is not None:
        nd_nue_sample = Sample(
            nd_nue_binning,
            [
                SubSample(
                    label="nd nuebar" if args.antinu else "nd nue",
                    initial_flavour=NuFlavour.electron,
                    final_flavour=NuFlavour.electron,
                    target_material=target_material,
                    antineutrino=args.antinu,
                ).fill_from_file(
                    file=NuisanceFile(args.nd_nue),
                    progress_bar=True,
                    max_n_events=args.max_events_per_file,
                )
            ],
            nd_parameters,
            name="ND nuebar" if args.antinu else "ND nue",
            ## now apply selection
        ).apply_selection(nd_nue_selection, progress_bar=True)

        print(
            f"events in sample {nd_nue_sample.name}: {np.sum(nd_nue_sample.get_event_rates())}"
        )

        nd_nue_sample.to_file("nd_nuebar.nps" if args.antinu else "nd_nue.nps")

    if args.fd_numu_numu is not None:

        fd_numu_sample = Sample(
            fd_numu_binning,
            [
                SubSample(
                    label="fd numubar -> numubar " if args.antinu else "fd numu -> numu ",
                    initial_flavour=NuFlavour.muon,
                    final_flavour=NuFlavour.muon,
                    target_material=target_material,
                    oscillator=oscillator,
                    antineutrino=args.antinu,
                ).fill_from_file(
                    file=NuisanceFile(args.fd_numu_numu),
                    progress_bar=True,
                    max_n_events=args.max_events_per_file,
                )
            ],
            fd_parameters,
            name="FD numubar" if args.antinu else "FD numu",
            ## now apply selection
        ).apply_selection(fd_numu_selection, progress_bar=True)

        print(
            f"events in sample {fd_numu_sample.name}: {np.sum(fd_numu_sample.get_event_rates())}"
        )

        if args.smear_energies:

            fd_energy_smear.apply(fd_numu_sample)

        fisher_info_calculator.apply(fd_numu_sample)
        fd_numu_sample.to_file("fd_numubar.nps" if args.antinu else "fd_numu.nps")

    if args.fd_numu_nue is not None and args.fd_nue_nue is not None:

        fd_nue_sample = Sample(
            fd_nue_binning,
            [
                SubSample(
                    label="fd nuebar -> nuebar " if args.antinu else "fd nue -> nue",
                    initial_flavour=NuFlavour.electron,
                    final_flavour=NuFlavour.electron,
                    target_material=target_material,
                    oscillator=oscillator,
                    antineutrino=args.antinu,
                ).fill_from_file(
                    file=NuisanceFile(args.fd_nue_nue),
                    progress_bar=True,
                    max_n_events=args.max_events_per_file,
                ),
                SubSample(
                    label="fd numubar -> nuebar " if args.antinu else "fd numubar -> nuebar",
                    initial_flavour=NuFlavour.muon,
                    final_flavour=NuFlavour.electron,
                    target_material=target_material,
                    oscillator=oscillator,
                    antineutrino=args.antinu,
                ).fill_from_file(
                    file=NuisanceFile(args.fd_numu_nue),
                    progress_bar=True,
                    max_n_events=args.max_events_per_file,
                ),
            ],
            fd_parameters,
            name="FD nuenar" if args.antinu else "FD nue",
            ## apply selection
        ).apply_selection(fd_nue_selection, progress_bar=True)

        print(
            f"events in sample {fd_nue_sample.name}: {np.sum(fd_nue_sample.get_event_rates())}"
        )

        if args.smear_energies:

            fd_energy_smear.apply(fd_nue_sample)

        fisher_info_calculator.apply(fd_nue_sample)
        fd_nue_sample.to_file("fd_nuebar.nps" if args.antinu else "fd_nue.nps")

    fisher_info_calculator.finalise()


if __name__ == "__main__":
    main()
