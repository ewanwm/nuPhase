import abc
import typing

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np

from nuPhase.sample import Sample, Binning
from nuPhase.oscillator import OscillationCalculator
from nuPhase.utils import strip_file_extension
from nuPhase.modes import cc_modes, nc_modes
from nuPhase.modules.base import AnalysisBase

class UnconstrainableNueAnalysis(AnalysisBase):

    def __init__(
        self,
        out_file_name: str,
        interaction_space: Binning,
        nd_samples: typing.List[Sample],
        fd_samples: typing.List[Sample]
    ):

        self.nd_samples: typing.List[Sample] = nd_samples
        self.fd_samples: typing.List[Sample] = fd_samples

        self.interaction_space: Binning = interaction_space

        self._pdf = PdfPages(out_file_name)

    def run(self):

        for sample in [*self.nd_samples, *self.fd_samples]:
            if sample is not None:

                fig, ax = plt.subplots()

                sample.imshow(
                    ax,
                    binning=self.interaction_space,
                    data_override=sample.get_event_rates(
                        binning=self.interaction_space
                    ),
                )
                ax.set_title(f"{sample.name}")
                self._pdf.savefig(fig)

                fig.clear()

        for fd_sample in self.fd_samples:

            for nd_sample in self.nd_samples:

                fig, ax = plt.subplots()

                fd_sample.imshow(
                    ax,
                    binning=self.interaction_space,
                    data_override=self.get_unconstrained(
                        nd_sample=nd_sample, fd_sample=fd_sample
                    ),
                )
                ax.set_title(f"{fd_sample.name} Unconstrained by\n{nd_sample.name}")
                self._pdf.savefig(fig)

        self._pdf.close()

    def get_unconstrained(self, nd_sample: Sample, fd_sample: Sample) -> np.ndarray:

        nd_event_rate = nd_sample.get_event_rates(binning=self.interaction_space)
        fd_event_rate = fd_sample.get_event_rates(binning=self.interaction_space)

        fd_event_rate[(nd_event_rate >= 1)] = 0

        return fd_event_rate


class BasicAnalysis(AnalysisBase):

    def __init__(self, out_file_name: str, samples: typing.List[Sample]):

        self.samples = samples

        ## open up a pdf to put plots in
        self._pdf = PdfPages(out_file_name)

    def run(self):

        ## make plots of the flux of each sample
        for sample in self.samples:
            self.make_flux_plots(sample)

        ## make plots of the Xsecs for each sample
        for sample in self.samples:
            self.make_xsec_plots(sample)

        ## make plots of the event rates for each sample
        for sample in self.samples:
            for binning_1d in [
                sample.binning.project([var]) for var in sample.binning.variables
            ]:
                self.make_1d_rate_plots(
                    sample, binning=binning_1d, cumulative=True, fill=True
                )

        ## make plots of event rates in for each sample using their own binning
        for sample in self.samples:

            if sample.binning.n_dims == 2:
                self.make_sample_binning_rate_plots(sample, 1.0)

        self._pdf.close()

    def make_sample_binning_rate_plots(
        self, sample: Sample, min_n_events: float = -np.inf
    ):

        fig, ax = plt.subplots()

        event_rate = sample.get_event_rates()

        ## locations where expected n of events is at least 1
        event_rate *= event_rate > min_n_events

        sample.imshow(ax, data_override=event_rate)
        self._pdf.savefig(fig)

        plt.close(fig)

    def make_flux_plots(self, sample: Sample):

        if sample.subsamples is not None:

            fig = plt.figure()

            for subsample in sample.subsamples:

                count, bin_edges = subsample.flux_hist
                bin_widths = bin_edges[1:] - bin_edges[:-1]

                count /= bin_widths / 0.05

                plt.stairs(count, bin_edges, label=subsample.name)

            plt.legend()
            plt.xlabel("neutrino energy [GeV]")
            plt.yscale("log")
            plt.title(f"{sample.name} Flux")
            plt.ylabel(f"Flux [1 / (cm^2 * 50 MeV * 10^21 POT)]")
            self._pdf.savefig(fig)

            plt.close(fig)

        else:
            pass

    def make_xsec_plots(self, sample: Sample):

        if sample.subsamples is not None:

            for subsample in sample.subsamples:

                fig = plt.figure()

                all_cc_codes = []

                for mode, codes in zip(
                    list(cc_modes.keys())[::-1], list(cc_modes.values())[::-1]
                ):

                    all_cc_codes += codes

                    enu = subsample.get_array(
                        "Enu_true", cut=lambda event: abs(event.mode) in codes
                    )

                    flux_bins = subsample.flux_binning.bin_edges[0]
                    bin_widths = flux_bins[1:] - flux_bins[:-1]

                    xsec = (
                        np.histogram(enu, bins=flux_bins)[0] / bin_widths
                        * subsample.get_xsec_weight() * subsample.get_integrated_flux() / subsample.flux_hist[0]
                    )

                    ## make basic flux plot
                    plt.stairs(xsec, subsample.flux_binning.bin_edges[0], label=mode)

                    plt.xscale("log")

                plt.legend()
                plt.xlabel(f"Enu_true")
                plt.title(f"{subsample.name} Xsec")
                plt.ylabel(f"XSec [1 / cm^2 / Nucleon]")
                self._pdf.savefig(fig)

                plt.close(fig)

        else:
            pass

    def make_1d_rate_plots(
        self,
        sample: Sample,
        binning: Binning,
        cumulative: bool = False,
        logy: bool = False,
        **stairs_args,
    ):

        assert binning.n_dims == 1, "Can only do 1D rate plots for 1D binning.... duh"

        fig = plt.figure()

        event_rate = sample.get_event_rates(binning=binning)
        plt.stairs(event_rate, binning.bin_edges[0], label="total")

        event_rate[:] = 0.0
        mode_event_rates = []

        all_cc_codes = []
        if cumulative:

            for mode, codes in zip(
                list(cc_modes.keys())[::-1], list(cc_modes.values())[::-1]
            ):

                all_cc_codes += codes

                event_rate += sample.get_event_rates(
                    cut=lambda event: abs(event.mode) in codes, binning=binning
                )
                mode_event_rates.append(np.copy(event_rate))

            ## add "other"
            event_rate += sample.get_event_rates(
                cut=lambda event: not abs(event.mode) in all_cc_codes, binning=binning
            )
            mode_event_rates.append(np.copy(event_rate))

            for mode_event_rate, mode in zip(
                mode_event_rates[::-1], ["other", *cc_modes.keys()]
            ):

                plt.stairs(mode_event_rate, binning.bin_edges[0], label=mode, **stairs_args)

        else:
            for mode, codes in zip(list(cc_modes.keys()), list(cc_modes.values())):

                plt.stairs(
                    sample.get_event_rates(
                        cut=lambda event: abs(event.mode) in codes, binning=binning
                    ),
                    binning.bin_edges[0],
                    label=mode,
                    **stairs_args,
                )

        if logy:
            plt.yscale("log")

        plt.legend()
        plt.xlabel(f"{binning.variables[0]}")
        plt.title(f"Event rate {sample.name}")
        plt.ylabel(
            f"N Events"
        )
        self._pdf.savefig(fig)

        plt.close(fig)


class FisherInfoAnalysis(AnalysisBase):

    def __init__(
        self,
        out_file_name: str,
        nd_samples: typing.List[Sample],
        fd_samples: typing.List[Sample],
        interaction_space: Binning,
    ):

        self._pdf = PdfPages(out_file_name)
        self._map_pdf = PdfPages(
            strip_file_extension(out_file_name, "pdf") + "-fisher-info-map.pdf"
        )
        self._per_event_map_pdf = PdfPages(
            strip_file_extension(out_file_name, "pdf")
            + "-per-event-fisher-info-by-energy.pdf"
        )
        self._fig = plt.figure()

        self.nd_samples = nd_samples
        self.fd_samples = fd_samples

        self.interaction_space = interaction_space

        ## set up fisher information maps
        self._fisher_info_maps = [ self.make_fisher_info_map(sample=sample) for sample in self.fd_samples ]

    def run(self):

        for nd_sample in self.nd_samples:

            self.do_fisher_info_projection(sample=nd_sample, make_plots=True)

        self._pdf.close()
        self._map_pdf.close()
        self._per_event_map_pdf.close()

    def do_fisher_info_projection(self, sample: Sample, make_plots: bool = False):

        for parameter in OscillationCalculator.parameter_names:

            for event in sample.events:

                variables = self.interaction_space.variables
                interaction_vars = []

                for i_var in range(self.interaction_space.n_dims):
                    interaction_vars.append(event.get_var(variables[i_var]))

                interaction_bins = tuple(
                    self.interaction_space.digitize(interaction_vars)
                )

                try:

                    fisher_info = 0.0

                    for map in self._fisher_info_maps:

                        fisher_info += map[parameter][interaction_bins]

                    event.aux_vars[f"{parameter}_projected_fisher_info"] = fisher_info

                except IndexError:

                    event.aux_vars[f"{parameter}_projected_fisher_info"] = 0.0
                    continue

            if make_plots:

                for binning in sample.binning.get_2d_projections():

                    fisher_weighted_event_rate = sample.get_event_rates(
                        binning=binning,
                        weight_var=f"{parameter}_projected_fisher_info",
                        keep_zero=True,
                    )
                    event_rate = sample.get_event_rates(
                        keep_zero=True,
                        binning=binning,
                    )

                    fig, ax = plt.subplots()

                    sample.imshow(
                        ax,
                        fisher_weighted_event_rate,
                        z_label="Projected Fisher Info",
                        binning=binning,
                    )
                    plt.title(f"{sample.name} {parameter} \nProjected Fisher Info")

                    self._pdf.savefig(fig)
                    fig.clear()
                    ax = fig.subplots()

                    sample.imshow(
                        ax,
                        fisher_weighted_event_rate / event_rate,
                        z_label="Mean Projected Fisher Info",
                        binning=binning,
                    )
                    plt.title(
                        f"{sample.name} {parameter} \nProjected Per-event Fisher Info"
                    )
                    self._pdf.savefig(fig)

                    plt.close(fig)

    def plot_fisher_info_map(
        self, variables: typing.List[str], slice_var: str = None, avg_per_event=False
    ):

        for sample, fisher_info_map in zip(
            self.fd_samples,
            self._fisher_info_maps,
        ):

            if sample is None:
                continue

            ## the projection from the user specified variables
            projection_binning = Binning(
                variables=variables,
                bins=[
                    self.interaction_space.get_bin_edges(variables[0]),
                    self.interaction_space.get_bin_edges(variables[1]),
                ],
            )

            ## get the indices that need to be summed over
            ## remove the user specified variable indices so we should be left with only
            ## ones we wanna get rid of
            sum_dim_indices = list(range(len(self.interaction_space.variables)))
            if slice_var is not None:
                sum_dim_indices.remove(
                    self.interaction_space.variables.index(slice_var)
                )
            for _var in variables:
                sum_dim_indices.remove(self.interaction_space.variables.index(_var))

            ## if we're averaging over events, pre-calculate the norm factor
            norm_factor = 1.0
            if avg_per_event:
                norm_factor = sample.get_event_rates(binning=projection_binning)

            for parameter_name in OscillationCalculator.parameter_names:

                slice_iterator = None
                slice_bins = None
                if slice_var is None:
                    slice_iterator = [0]
                else:
                    slice_iterator = range(self.interaction_space.get_n_bins(slice_var))
                    slice_bins = self.interaction_space.get_bin_edges(slice_var)

                for i_slice in slice_iterator:

                    fig, ax = plt.subplots()

                    dat = fisher_info_map[parameter_name]

                    min_info = np.min(dat / norm_factor)
                    max_info = np.max(dat / norm_factor)

                    if len(sum_dim_indices) > 0:
                        dat = np.sum(dat, axis=sum_dim_indices, keepdims=True)

                    if slice_var is not None:
                        dat = np.take(
                            dat / norm_factor,
                            i_slice,
                            axis=self.interaction_space.variables.index(slice_var),
                        )

                    ## now that we have summed and sliced we should be left with only the spcified binning variables

                    ## squeeze to get rid of any lingering size 1 summed over dimensions
                    dat = np.squeeze(dat)

                    ## this makes the plot a bit easier to read
                    dat[dat == 0.0] = np.nan

                    z_label = f"Fisher Info{' / N Events' if avg_per_event else ''}"

                    sample.imshow(
                        ax,
                        dat,
                        binning=projection_binning,
                        z_label=z_label,
                        vmin=min_info,
                        vmax=max_info,
                    )
                    plt.xlabel(variables[0])
                    plt.ylabel(variables[1])

                    if slice_var is not None:
                        plt.title(
                            f"{sample.name} {parameter_name} \nFisher Info{' Per Event' if avg_per_event else ''} \n{slice_bins[i_slice]} < {slice_var} < {slice_bins[i_slice + 1]} GeV"
                        )

                    else:
                        plt.title(
                            f"{sample.name} {parameter_name} \nFisher Info{' Per Event' if avg_per_event else ''}"
                        )

                    self._map_pdf.savefig(fig)
                    plt.close(fig)

    def make_fisher_info_map(self, sample: Sample):

        fisher_info_map = {}

        for parameter_name in OscillationCalculator.parameter_names:

            fisher_info_map[parameter_name] = sample.get_event_rates(
                binning=self.interaction_space,
                weight_var=f"{parameter_name}_fisher_info",
            )

        return fisher_info_map
