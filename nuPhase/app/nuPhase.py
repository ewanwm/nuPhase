from nuPhase.utils import strip_file_extension
from nuPhase.sample import Sample, Binning, SubSample, SubSampleParameters, flavour_from_name, NuisanceFile
from nuPhase.modules.analysis import (
    FisherInfoAnalysis,
    BasicAnalysis,
    UnconstrainableNueAnalysis,
)
from nuPhase.oscillator import OscillationCalculator

from nuPhase.materials import material_from_name
from nuPhase.modules.module_list import moduleTypeEnum, ModuleList

from argparse import ArgumentParser, HelpFormatter
import sys

import numpy as np

flux_bins = np.array(
    [
        0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.3,
        0.31,
        0.32,
        0.33,
        0.34,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.41,
        0.42,
        0.43,
        0.44,
        0.45,
        0.46,
        0.47,
        0.48,
        0.49,
        0.5,
        0.51,
        0.52,
        0.53,
        0.54,
        0.55,
        0.56,
        0.57,
        0.58,
        0.59,
        0.6,
        0.61,
        0.62,
        0.63,
        0.64,
        0.65,
        0.66,
        0.67,
        0.68,
        0.69,
        0.7,
        0.71,
        0.72,
        0.73,
        0.74,
        0.75,
        0.76,
        0.77,
        0.78,
        0.79,
        0.8,
        0.82,
        0.84,
        0.86,
        0.88,
        0.9,
        0.95,
        1,
        1.05,
        1.1,
        1.15,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.6,
        2.8,
        3,
        3.2,
        3.4,
        3.6,
        3.8,
        4,
        4.5,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
)


def setup_parser():

    parser = ArgumentParser("make-plots",
        formatter_class=lambda prog: HelpFormatter(prog,max_help_position=40))

    parser.add_argument(
        "-o", "--output", type=str, help="name of output file", required=True
    )

    ## set up subcommand parsers
    subparsers = parser.add_subparsers(title = "Commands", required=True, dest="command")

    ## set up subsample maker command
    prepare_subsample_parser = subparsers.add_parser("prepare-subsample", help="'prepare' a subsample - Convert it from nuisance flattree into a nuPhase object that canbe used for further analysis")
    prepare_subsample_parser.set_defaults(func = prepare_subsample)
    prepare_subsample_parser.add_argument('--nuisance-file', help="The name of the input file describing the MC events", required=True, type=str)
    prepare_subsample_parser.add_argument("--oscillator-baseline", help="The baseline of this subsample. If None then no oscillations will be applied", type=float, required=False, default=None)
    prepare_subsample_parser.add_argument("--oscillator-density", help="The density of the propagation medium for this subsample", type=float, required=False, default=2.6)
    prepare_subsample_parser.add_argument('--base-pot', help="The POT that was assumed when generating this subsample", required=True, type=float)
    prepare_subsample_parser.add_argument('--target-material', help="The target material that this subsample was generated with", required=True, type=str)
    prepare_subsample_parser.add_argument('--target-mass', help="The target mass to scale to", required=True, type=float)
    prepare_subsample_parser.add_argument('--initial-flavour', help="The initial (unoscillated) neutrino flavour", required=True, type=str)
    prepare_subsample_parser.add_argument('--final-flavour', help="The final (oscillated) neutrino flavour", required=True, type=str)
    prepare_subsample_parser.add_argument('--name', help="A name for this subsample", required=True, type=str)
    prepare_subsample_parser.add_argument('--antinu', help="The flag to declare that this subsample was generated for antinueutrino (RHC) mode", action="store_true", required=False)
    prepare_subsample_parser.add_argument('--target-pot', help="The number of POT to scale the sample to - if not specified then the base pot will be used", required=False, default=None, type=float)
    prepare_subsample_parser.add_argument('--max-n-events', "-n", help="Maximum number of events to read from the input file - if not specified then all will be read", required=False, default=None, type=int)

    ## set up sample maker command
    prepare_sample_parser = subparsers.add_parser("prepare-sample", help="'prepare' a sample - Combine subsamples into a single Sample object that can be passed to analysis modules")
    prepare_sample_parser.set_defaults(func = prepare_sample)
    prepare_sample_parser.add_argument('--subsamples', nargs='+', default=[], help="The name of the input file describing the MC events", required=True)
    prepare_sample_parser.add_argument('--binning', help="Path to config file defining the binning for the sample", required=True, type=str)
    prepare_sample_parser.add_argument('--name', help="Name for this sample", required=True, type=str)
      
    ## set up fisher information command
    fisher_info_parser = subparsers.add_parser("fisher-analysis", help="Perform Fisher information based analysis - will construct fisher info map from FD samples, propagate the info through to the nd samples")
    fisher_info_parser.set_defaults(func = fisher_analysis)
    fisher_info_parser.add_argument('--fd-samples', nargs='+', default=[], help="list of far detector samples to consider", required=True)
    fisher_info_parser.add_argument('--nd-samples', nargs='+', default=[], help="list of near detector samples to consider", required=True)

    ## set up basic analysis command
    basic_analysis_parser = subparsers.add_parser("basic-analysis", help="Perform basic analysis - make plots of the provided samples... that's it really")
    basic_analysis_parser.set_defaults(func = basic_analysis)
    basic_analysis_parser.add_argument('--samples', nargs='+', default=[], help="list of samples to consider", required=True)
    
    ## set up unconstrainable nue analysis command
    unconstrainable_analysis_parser = subparsers.add_parser("unconstrainable-events-analysis", help="Perform analysis to find events in far detector samples that are unconstrainable by ND samples")
    unconstrainable_analysis_parser.set_defaults(func = unconstrainable_analysis)
    unconstrainable_analysis_parser.add_argument('--fd-samples', nargs='+', default=[], help="list of far detector samples to consider", required=True)
    unconstrainable_analysis_parser.add_argument('--nd-samples', nargs='+', default=[], help="list of near detector samples to consider", required=True)
    
    ## set up parser for applying transform to a sample
    apply_transform_parser = subparsers.add_parser("apply-transformation", help="Apply some transformation to a sample",
        formatter_class=lambda prog: HelpFormatter(prog,max_help_position=40)
    )
    apply_transform_parser.set_defaults(func = apply_transformation)
    apply_transform_parser.add_argument("--strip-particle-info", action="store_true", help="Strip particle level information from the event. Saves space but won't be able to apply any more selections or transformations requiring particle level info")
    transformation_subparsers = apply_transform_parser.add_subparsers(title = "Transformations", dest="transformation")

    for transformation in ModuleList().get_selection_modules() + ModuleList().get_transformation_modules():
        module_instance = transformation()
        module_parser = transformation_subparsers.add_parser(transformation.__name__, help=module_instance.help())
        module_instance.setup_parser(module_parser)
        
    return parser

def apply_transformation(args, output_file):

    sample = Sample.from_file(args.input_sample)

    module = ModuleList().get_module(args.transformation)

    ## create an instance of the module class
    module_instance = module()
    module_instance.parse_args(args)

    if ModuleList().get_module_type(module) == moduleTypeEnum.transformation:
        module_instance.initialise(sample)
        sample.apply_transformation(transformation=module_instance, progress_bar=args.progress, strip_particle_info=args.strip_particle_info).to_file(output_file)
    elif ModuleList().get_module_type(module) == moduleTypeEnum.selection:
        module_instance.initialise(sample)
        sample.apply_selection(selection=module_instance, progress_bar=args.progress, strip_particle_info=args.strip_particle_info).to_file(output_file)
    else:
        raise ValueError(f"provided module ({args.transformation}) is not a transformation or selection :(")

    module_instance.finalise(sample)

def fisher_analysis(args, output_file):

    ## set up samples
    nd_samples = [ Sample.from_file(file_name) for file_name in args.nd_samples ]
    fd_samples = [ Sample.from_file(file_name) for file_name in args.fd_samples ]

    analysis = FisherInfoAnalysis(
        output_file + "-fisher-analysis.pdf",
        nd_samples=nd_samples,
        fd_samples=fd_samples,
        interaction_space=Binning(
            ["Enu_true", "q3", "q0"],
            bin_edges=[flux_bins, np.linspace(0, 2.0, 50), np.linspace(0, 2.0, 50)],
        )
    )

    analysis.plot_fisher_info_map(variables=["q3", "q0"], slice_var="Enu_true")
    analysis.plot_fisher_info_map(
        variables=["q3", "q0"], slice_var="Enu_true", avg_per_event=True
    )
    analysis.run()

def prepare_subsample(args, output_file):

    initial_flavour = flavour_from_name(args.initial_flavour)
    final_flavour = flavour_from_name(args.final_flavour)

    target_material = material_from_name(args.target_material)

    target_pot = args.target_pot
    if target_pot is None:
        target_pot = args.base_pot

    oscillator = None
    if args.oscillator_baseline is not None:
        oscillator = OscillationCalculator(baseline=args.oscillator_baseline, density=args.oscillator_density, initialisation="pdg")

    parameters = SubSampleParameters(
        pot=target_pot,
        target_material=target_material,
        target_mass=args.target_mass,
        initial_flavour=initial_flavour,
        final_flavour=final_flavour,
        antineutrino=args.antinu
    )
    subsample = SubSample(
        name=args.name,
        parameters=parameters,
        base_pot=args.base_pot,
        oscillator=oscillator,
    )

    ## fill it with the nuisance file
    nuisance_file = NuisanceFile(file_name=args.nuisance_file)
    subsample.fill_from_file(nuisance_file, progress_bar=True, max_n_events=args.max_n_events)

    ## save it to disk
    subsample.to_file(output_file)

def prepare_sample(args, output_file):

    ## create binning object
    binning = Binning.from_file(args.binning)

    ## create sample object
    Sample(
        binning=binning,
        subsamples=[SubSample.from_file(file_name) for file_name in args.subsamples],
        name=args.name
    ## save it to a nuPhase file
    ).to_file(output_file)

def basic_analysis(args, output_file):

    BasicAnalysis(
        output_file + "-basic-plots.pdf",
        samples=[ Sample.from_file(file_name) for file_name in args.samples ],
    ).run()

def unconstrainable_analysis(args, output_file):

    UnconstrainableNueAnalysis(
        output_file + "-unconstrainable.pdf",
        nd_samples=[ Sample.from_file(file_name) for file_name in args.nd_samples ],
        fd_samples=[ Sample.from_file(file_name) for file_name in args.fd_samples ],
        interaction_space=Binning(
            ["q3", "q0"], bins=[np.linspace(0, 2.0, 50), np.linspace(0, 2.0, 50)]
        ),
    ).run()


def main():

    parser = setup_parser()

    ## parse args
    args = parser.parse_args(sys.argv[1:])

    output_file: str = strip_file_extension(args.output, "pdf")

    ## run the relevant function
    args.func(args, output_file)

if __name__ == "__main__":
    main()
