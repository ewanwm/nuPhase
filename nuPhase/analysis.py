from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import typing

from nuTens import tensor
from nuTens.tensor import Tensor
from nuTens.autograd import grad

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

        fig, ax = plt.subplots()

        self.fd_nue.imshow(ax, data_override=self.get_unconstrained(nd_sample = self.nd_numu, fd_sample = self.fd_nue))
        ax.set_title("FD nue Unconstrained by ND Numu")
        self._pdf.savefig(fig)

        fig.clear()

        if self.nd_nue:

            ax = fig.subplots()
            self.fd_nue.imshow(ax, data_override=self.get_unconstrained(nd_sample = self.nd_nue, fd_sample = self.fd_nue))

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

        self.make_fisher_event_rates(sample = self.fd_nue, make_plots = True)
        self._nue_fisher_info_map = self.make_fisher_info_map(sample = self.fd_nue)

        if self.fd_numu is not None:
            self.make_fisher_event_rates(sample = self.fd_numu, make_plots = True)
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

    def make_fisher_event_rates(self, sample: Sample, make_plots: bool = True):

        _, _, fisher_info = self.get_fisher_info(sample = sample, binning = sample.binning, make_plots=True)
        event_rates = sample.get_event_rates( binning = sample.binning, keep_zero=True)

        for parameter_name in self.oscillator.parameters.keys():

            per_event_fisher_info = fisher_info[parameter_name] / event_rates
            per_event_fisher_info[event_rates <= 1.0] = 0.0

            for event in sample.events:

                bins = tuple(sample.binning.digitize([event.get_var(var) for var in sample.binning.variables]))

                try:
                    event.aux_vars[f"{parameter_name}_fisher_info"] = per_event_fisher_info[bins]
                except IndexError:
                    event.aux_vars[f"{parameter_name}_fisher_info"] = 0.0
                    continue

            if make_plots:

                if sample.binning.n_dims == 1:

                    fig, ax = plt.subplots()

                    dat = fisher_info[parameter_name][...]
                    dat[dat == 0.0] = np.nan

                    plt.stairs(dat, sample.binning.bins[0])
                    plt.xlabel(f"{sample.binning.variables[0]}")
                    plt.ylabel("Fisher info")
                    plt.title(f"{sample.name} {parameter_name} \nFisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)
                    fig, ax = plt.subplots()

                    dat = per_event_fisher_info[...]
                    dat[dat == 0.0] = np.nan

                    plt.stairs(dat, sample.binning.bins[0])
                    plt.xlabel(f"{sample.binning.variables[0]}")
                    plt.ylabel("Fisher info")
                    plt.title(f"{sample.name} {parameter_name} \nPer-event Fisher Information")
                    self._pdf.savefig(fig)

                    
                    plt.close(fig)
                    
                elif sample.binning.n_dims == 2:
                    fig, ax = plt.subplots()

                    dat = fisher_info[parameter_name][...]
                    dat[dat <= 1.0] = np.nan

                    sample.imshow(ax, data_override = dat, z_label = "Fisher Information")
                    plt.title(f"{sample.name} {parameter_name} \nFisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)
                    fig, ax = plt.subplots()

                    dat = per_event_fisher_info[...]
                    dat[dat <= 1.0] = np.nan

                    sample.imshow(ax, data_override = dat, z_label = "Per-event Fisher Information")
                    plt.title(f"{sample.name} {parameter_name} \nPer-event Fisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)

                else:
                    raise ValueError("Can't make fisher info plots for n-dims != 1 or 2 :(")

    def get_fisher_info(self, sample: Sample, binning: Binning = None, make_plots: bool = False) -> typing.Tuple[np.ndarray]:

        if binning is None:
            binning = sample.binning
            
        ## calculate the osc probs
        sample.oscillate_events(progress_bar = True)

        bin_contents = None
        if binning.n_dims == 1:
            bin_contents = [Tensor.zeros([1]).requires_grad(True) for _ in range(binning.n_bins[0] + 1)]
        
        elif binning.n_dims == 2:
            bin_contents = [[Tensor.zeros([1]).requires_grad(True) for j in range(binning.n_bins[1] + 1)] for i in range(binning.n_bins[0] + 1)]
            
        gradients = {}
        second_derivs = {}
        fisher_informations = {}
        for osc_par in self.oscillator.parameters.keys():
            gradients[osc_par] = np.zeros(binning.n_bins)
            second_derivs[osc_par] = np.zeros(binning.n_bins)
            fisher_informations[osc_par] = np.zeros(binning.n_bins)
            
        for subsample in sample.subsamples:

            for event in tqdm(subsample.events, desc="gettin' bin contents"):
                
                u_var = event.get_var(binning.variables[0])
                if u_var <= binning.bins[0][0] or u_var >= binning.bins[0][-1]:
                    continue

                u = np.digitize(u_var, binning.bins[0])
                v = None

                if binning.n_dims == 2:

                    v_var = event.get_var(binning.variables[1])
                    if v_var <= binning.bins[1][0] or v_var >= binning.bins[1][-1]:
                        continue
                    
                    v = np.digitize(event.get_var(binning.variables[1]), binning.bins[1])

                    bin_contents[u][v] = bin_contents[u][v] + event.get_var("osc_weight") * subsample.get_event_scaling(sample.parameters.target_mass, sample.parameters.pot)

                else:
                    bin_contents[u] = bin_contents[u] + event.get_var("osc_weight") * subsample.get_event_scaling(sample.parameters.target_mass, sample.parameters.pot)

        for i_bin in tqdm(range(1, binning.n_bins[0] + 1), desc = "gettin' gradients"):

            j_iterator = [-999]
            if binning.n_dims == 2:
                j_iterator = range(1, binning.n_bins[1] + 1)

            for j_bin in j_iterator:

                ## fill the gradient histogram for each osc parameter
                for osc_par_name, osc_par in zip(self.oscillator.parameters.keys(), self.oscillator.parameters.values()): 
                
                    ## do the backward propagation to get gradient in terms of each osc parameter
                    if binning.n_dims == 1:
                        if bin_contents[i_bin ].numpy()[0] == 0.0:
                            continue

                        gradient = grad(bin_contents[i_bin], osc_par)
                        second_deriv = grad(gradient, osc_par)
                        fisher_info = tensor.pow(grad(tensor.log(bin_contents[i_bin]), osc_par), 2.0)

                    elif binning.n_dims == 2:
                        if bin_contents[i_bin ][j_bin ].numpy()[0] == 0.0:
                            continue

                        gradient = grad(bin_contents[i_bin][j_bin], osc_par)
                        second_deriv = grad(gradient, osc_par)
                        fisher_info = tensor.pow(grad(tensor.log(bin_contents[i_bin][j_bin]), osc_par), 2.0)

                    if binning.n_dims == 1:
                        gradients[osc_par_name][i_bin - 1] = gradient.numpy()[0]
                        second_derivs[osc_par_name][i_bin - 1] = second_deriv.numpy()[0]
                        fisher_informations[osc_par_name][i_bin - 1] = fisher_info.numpy()[0]
                    elif binning.n_dims == 2:
                        gradients[osc_par_name][i_bin - 1][j_bin -1] = gradient.numpy()[0]
                        second_derivs[osc_par_name][i_bin - 1][j_bin - 1] = fisher_info.numpy()[0]
                        fisher_informations[osc_par_name][i_bin - 1][j_bin - 1] = fisher_info.numpy()[0]

        if make_plots:
            for osc_par in self.oscillator.parameters.keys():

                for data_dict, label in zip([gradients, second_derivs, fisher_informations], ["Gradient", "2nd Deriv", "~Fisher Information"]):
                    fig, ax = plt.subplots()
                    
                    data = data_dict[osc_par]

                    data[data == 0] = np.nan

                    if binning.n_dims == 2:
                        u_bins, v_bins = binning.bins

                        mappable = ax.imshow(data.T, extent=(u_bins[0], u_bins[-1], v_bins[0], v_bins[-1]), origin="lower", aspect = "auto")

                        ax.set_xlabel(binning.variables[0])
                        ax.set_ylabel(binning.variables[1])

                        cbar = plt.colorbar(mappable)
                        cbar.set_label(f"{label}")
                        
                    elif binning.n_dims == 1:

                        ax.stairs(data, binning.bins[0])
                        ax.set_xlabel(binning.variables[0])
                        ax.set_ylabel(f"{label}")

                    plt.title(f"{sample.name} {osc_par} {label}")

                    self._pdf.savefig(fig)
                    
                    plt.close(fig)

        return gradients, second_derivs, fisher_informations