from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

import typing

from nuPhase.sample import Sample, Binning
from nuPhase.oscillator import OscillationCalculator
from nuPhase.utils import strip_file_extension
from nuPhase.modes import modes

class UnconstrainableNueAnalysis:

    def __init__(
            self,
            out_file_name: str,
            interaction_space: Binning,
            nd_numu: Sample,
            fd_nue: Sample, 
            nd_nue: Sample = None,
    ):
        
        binning_check = nd_numu.binning == fd_nue.binning
        if nd_nue is not None:
            binning_check = binning_check and nd_numu.binning == nd_nue.binning

        self.nd_numu: Sample = nd_numu
        self.nd_nue: Sample  = nd_nue
        self.fd_nue: Sample  = fd_nue

        self.interaction_space: Binning = interaction_space

        self._pdf = PdfPages(out_file_name)

    def run(self):

        for sample in [self.nd_numu, self.nd_nue, self.fd_nue]:
            if sample is not None:
                
                fig, ax = plt.subplots()

                sample.imshow(ax, binning = self.interaction_space, data_override = sample.get_event_rates(binning = self.interaction_space))
                ax.set_title(f"{sample.name}")
                self._pdf.savefig(fig)

                fig.clear()

        fig, ax = plt.subplots()

        self.fd_nue.imshow(ax, binning = self.interaction_space, data_override = self.get_unconstrained(nd_sample = self.nd_numu, fd_sample = self.fd_nue))
        ax.set_title("FD nue Unconstrained by ND Numu")
        self._pdf.savefig(fig)

        fig.clear()

        if self.nd_nue:

            ax = fig.subplots()
            self.fd_nue.imshow(ax, binning = self.interaction_space, data_override = self.get_unconstrained(nd_sample = self.nd_nue, fd_sample = self.fd_nue))

            ax.set_title("FD nue Unconstrained by ND Nue")
            self._pdf.savefig(fig)

        self._pdf.close()


    def get_unconstrained(self, nd_sample: Sample, fd_sample: Sample) -> np.ndarray:

        nd_event_rate = nd_sample.get_event_rates(binning = self.interaction_space)
        fd_event_rate = fd_sample.get_event_rates(binning = self.interaction_space)

        fd_event_rate[(nd_event_rate >= 1)] = 0

        return fd_event_rate

class BasicAnalysis:

    def __init__(
        self,
        out_file_name: str, 
        samples: typing.List[Sample]
    ):

        self.samples = samples

        ## open up a pdf to put plots in
        self._pdf = PdfPages(out_file_name)

    def run(self):

        ## make plots of the flux of each sample
        for sample in self.samples:
            self.make_flux_plots(sample)

        ## make plots of the Xsecs for each sample
        for sample in self.samples:
            for binning_1d in [sample.binning.project([var]) for var in sample.binning.variables]:
                self.make_xsec_plots(sample, binning_1d)

        ## make plots of the event rates for each sample
        for sample in self.samples:
            for binning_1d in [sample.binning.project([var]) for var in sample.binning.variables]:
                self.make_1d_rate_plots(sample, binning = binning_1d, cumulative=True, fill=True)

        ## make plots of event rates in for each sample using their own binning
        for sample in self.samples:

            self.make_sample_binning_rate_plots(sample, 1.0)

        self._pdf.close()

    def make_sample_binning_rate_plots(self, sample: Sample, min_n_events: float = -np.inf):

        fig, ax = plt.subplots()

        event_rate = sample.get_event_rates()

        ## locations where expected n of events is at least 1
        event_rate *= event_rate > min_n_events

        sample.imshow(ax, data_override = event_rate)
        self._pdf.savefig(fig)

        plt.close(fig)

    def make_flux_plots(self, sample: Sample):

        if sample.subsamples is not None:

            fig = plt.figure()

            for subsample in sample.subsamples:

                count, bin_edges = subsample.flux_hist

                plt.stairs(count, bin_edges, label = subsample.label)

            plt.legend()
            plt.xlabel("neutrino energy [GeV]")
            plt.yscale("log")
            plt.title(f"{sample.name} Flux")
            plt.ylabel(f"Flux [1 / (cm^2 * 50 MeV * 10^21 POT)]")
            self._pdf.savefig(fig)

            plt.close(fig)

        else:
            pass

    def make_xsec_plots(self, sample: Sample, binning: Binning):

        assert binning.n_dims == 1, "Can only make xsec plots for 1D binning :("
        if sample.subsamples is not None:

            for subsample in sample.subsamples:

                fig = plt.figure()

                for mode, codes in zip(list(modes.keys())[::-1], list(modes.values())[::-1]):

                    enu = subsample.get_array("Enu_true", cut = lambda event: event.mode in codes)

                    xsec = np.histogram(enu, bins = binning.bins[0])[0] * subsample.get_xsec_weight()

                    ## make basic flux plot
                    plt.stairs(xsec, binning.bins[0], label = mode)

                plt.legend()
                plt.xlabel(f"{binning.variables[0]}")
                plt.title(f"{subsample.label} Xsec")
                plt.ylabel(f"XSec [1 / cm^2 / Nucleon]")
                self._pdf.savefig(fig)

                plt.close(fig)

        else:
            pass

    def make_1d_rate_plots(self, sample: Sample, binning: Binning, cumulative: bool = False, logy: bool = False, **stairs_args):

        assert binning.n_dims == 1, "Can only do 1D rate plots for 1D binning.... duh"

        fig = plt.figure()

        event_rate = sample.get_event_rates(binning=binning)
        plt.stairs(event_rate, binning.bins[0], label="total")

        event_rate[:] = 0.0
        mode_event_rates = []

        if cumulative: 
            for mode, codes in zip(list(modes.keys())[::-1], list(modes.values())[::-1]):

                event_rate += sample.get_event_rates(cut = lambda event: event.mode in codes, binning=binning)
                mode_event_rates.append(np.copy(event_rate))
            
            for mode_event_rate, mode in zip(mode_event_rates[::-1], modes.keys()):

                plt.stairs(mode_event_rate, binning.bins[0], label=mode, **stairs_args)

        else:
            for mode, codes in zip(list(modes.keys()), list(modes.values())):

                plt.stairs(sample.get_event_rates(cut = lambda event: event.mode in codes, binning=binning), binning.bins[0], label=mode, **stairs_args)

        if logy:
            plt.yscale("log")
    
        plt.legend()
        plt.xlabel(f"{binning.variables[0]}")
        plt.title(f"Event rate {sample.name}")
        plt.ylabel(f"N Events / {sample.parameters.pot:.2E} POT / {sample.parameters.target_mass:.2E}")
        self._pdf.savefig(fig)

        plt.close(fig)
        
class FisherInfoAnalysis:

    def __init__(
            self, 
            out_file_name: str, 
            nd_numu: Sample, fd_nue: Sample, 
            oscillator: OscillationCalculator,
            interaction_space: Binning,
            nd_nue: Sample = None, fd_numu: Sample = None
        ):

        self._pdf = PdfPages(out_file_name)
        self._map_pdf = PdfPages(strip_file_extension(out_file_name, "pdf") + "-fisher-info-by-energy.pdf")
        self._per_event_map_pdf = PdfPages(strip_file_extension(out_file_name, "pdf") + "-per-event-fisher-info-by-energy.pdf")
        self._fig = plt.figure()

        self.nd_numu = nd_numu
        self.nd_nue = nd_nue
        self.fd_numu = fd_numu
        self.fd_nue = fd_nue

        self.interaction_space = interaction_space

        self.oscillator: OscillationCalculator = oscillator

        ## set up fisher information maps
        self._nue_fisher_info_map  = None
        self._numu_fisher_info_map = None

        self._nue_fisher_info_map = self.make_fisher_info_map(sample = self.fd_nue)

        if self.fd_numu is not None:
            self._numu_fisher_info_map = self.make_fisher_info_map(sample = self.fd_numu)

            
    def run(self):

        self.do_fisher_info_projection(sample = self.nd_numu, make_plots = True)

        if self.nd_nue is not None:
            self.do_fisher_info_projection(sample = self.nd_nue, make_plots = True)

        self._pdf.close()
        self._map_pdf.close()
        self._per_event_map_pdf.close()

    def do_fisher_info_projection(self, sample: Sample, make_plots: bool = False):

        for parameter in self.oscillator.parameters.keys():

            for event in sample.events:

                variables = self.interaction_space.variables
                interaction_vars = []

                for i_var in range(self.interaction_space.n_dims):
                    interaction_vars.append(event.get_var(variables[i_var]))

                interaction_bins = tuple(self.interaction_space.digitize(interaction_vars))

                try:

                    fisher_info = self._nue_fisher_info_map[parameter][interaction_bins]
                    event.aux_vars[f"{parameter}_projected_fisher_info"] = fisher_info

                except IndexError:

                    event.aux_vars[f"{parameter}_projected_fisher_info"] = 0.0
                    continue

            fisher_weighted_event_rate = sample.get_event_rates(weight_var = f"{parameter}_projected_fisher_info", keep_zero=True)
            event_rate = sample.get_event_rates(keep_zero=True)

            if make_plots:

                fig, ax = plt.subplots()
                        
                sample.imshow(ax, fisher_weighted_event_rate, z_label="Projected Fisher Info")
                plt.title(f"{sample.name} {parameter} \nProjected Fisher Info")

                self._pdf.savefig(fig)
                fig.clear()
                ax = fig.subplots()

                sample.imshow(ax, fisher_weighted_event_rate / event_rate, z_label="Mean Projected Fisher Info")
                plt.title(f"{sample.name} {parameter} \nProjected Per-event Fisher Info")
                self._pdf.savefig(fig)

                plt.close(fig)

    def plot_fisher_info_map(self, variables: typing.List[str], slice_var: str = None, avg_per_event = False):

        for sample, fisher_info_map in zip(
            [self.fd_nue, self.fd_numu],
            [self._nue_fisher_info_map, self._numu_fisher_info_map]
        ):

            if sample is None:
                continue

            ## the projection from the user specified variables
            projection_binning = Binning(variables = variables, bins = [self.interaction_space.get_bin_edges(variables[0]), self.interaction_space.get_bin_edges(variables[1])])

            ## get the indices that need to be summed over 
            ## remove the user specified variable indices so we should be left with only
            ## ones we wanna get rid of
            sum_dim_indices = list(range(len(self.interaction_space.variables)))
            if slice_var is not None:
                sum_dim_indices.remove(self.interaction_space.variables.index(slice_var))
            for _var in variables:
                sum_dim_indices.remove(self.interaction_space.variables.index(_var))

            ## if we're averaging over events, pre-calculate the norm factor
            norm_factor = 1.0
            if avg_per_event:
                norm_factor = sample.get_event_rates(binning = projection_binning)

            for parameter_name in self.oscillator.parameters.keys():

                slice_iterator = None
                slice_bins     = None
                if slice_var is None:
                    slice_iterator = [0]
                else:
                    slice_iterator = range(self.interaction_space.get_n_bins(slice_var))
                    slice_bins = self.interaction_space.get_bin_edges(slice_var)

                for i_slice in slice_iterator: 

                    fig, ax = plt.subplots()

                    dat = fisher_info_map[parameter_name]

                    if len(sum_dim_indices) > 0:
                        dat = np.sum(dat, axis = sum_dim_indices, keepdims = True)

                    if slice_var is not None:
                        dat = np.take(dat / norm_factor, i_slice, axis = self.interaction_space.variables.index(slice_var))

                    ## now that we have summed and sliced we should be left with only the spcified binning variables
                    
                    ## squeeze to get rid of any lingering size 1 summed over dimensions
                    dat = np.squeeze(dat)

                    ## this makes the plot a bit easier to read
                    dat[dat == 0.0] = np.nan

                    z_label = f"Fisher Info{' / N Events' if avg_per_event else ''}"

                    sample.imshow(ax, dat, binning = projection_binning, z_label = z_label)
                    plt.xlabel(variables[0])
                    plt.ylabel(variables[1])
                    
                    if slice_var is not None:
                        plt.title(f"{sample.name} {parameter_name} \nFisher Info{' Per Event' if avg_per_event else ''} \n{slice_bins[i_slice]} < {slice_var} < {slice_bins[i_slice + 1]} GeV")

                    else:
                        plt.title(f"{sample.name} {parameter_name} \nFisher Info{' Per Event' if avg_per_event else ''}")

                    self._map_pdf.savefig(fig)
                    plt.close(fig)
            
    def make_fisher_info_map(self, sample: Sample):

        fisher_info_map = {}

        for parameter_name in self.oscillator.parameters.keys():

            fisher_info_map[parameter_name] = sample.get_event_rates(binning = self.interaction_space, weight_var = f"{parameter_name}_fisher_info")

        return fisher_info_map
