import typing

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm

from nuTens import tensor
from nuTens.tensor import Tensor
from nuTens.autograd import grad

from nuPhase.sample import Sample, Binning
from nuPhase.oscillator import OscillationCalculator

class CalculateFisherInfo:

    def __init__(self, oscillator: OscillationCalculator, make_plots: bool = True, plot_file_name: str = "Fisher-info.pdf"):
        
        self.make_plots = make_plots
        self.oscillator = oscillator

        if self.make_plots:

            self._pdf = PdfPages(plot_file_name)

    def finalise(self):

        self._pdf.close()

    def apply(self, sample: Sample):

        self.make_fisher_event_rates(sample)

    def make_fisher_event_rates(self, sample: Sample):

        fisher_info = self.get_fisher_info(sample = sample, binning = sample.binning)
        event_rates = sample.get_event_rates( binning = sample.binning, keep_zero=True)

        for parameter_name in self.oscillator.parameters.keys():

            per_event_fisher_info = fisher_info[parameter_name] / event_rates

            for event in sample.events:

                bins = tuple(sample.binning.digitize([event.get_var(var) for var in sample.binning.variables]))

                try:
                    event.aux_vars[f"{parameter_name}_fisher_info"] = per_event_fisher_info[bins]
                except IndexError:
                    event.aux_vars[f"{parameter_name}_fisher_info"] = 0.0
                    continue

            if self.make_plots:

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
                    dat[dat == 0.0] = np.nan

                    sample.imshow(ax, data_override = dat, z_label = "Fisher Information")
                    plt.title(f"{sample.name} {parameter_name} \nFisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)
                    fig, ax = plt.subplots()

                    dat = per_event_fisher_info[...]
                    dat[dat == 0.0] = np.nan

                    sample.imshow(ax, data_override = dat, z_label = "Per-event Fisher Information")
                    plt.title(f"{sample.name} {parameter_name} \nPer-event Fisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)

                else:
                    raise ValueError("Can't make fisher info plots for n-dims != 1 or 2 :(")

    def get_fisher_info(self, sample: Sample, binning: Binning = None) -> typing.Tuple[np.ndarray]:

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
        fisher_informations = {}
        for osc_par in self.oscillator.parameters.keys():
            gradients[osc_par] = np.zeros(binning.n_bins)
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
                        fisher_info = tensor.pow(gradient, 2.0)

                    elif binning.n_dims == 2:
                        if bin_contents[i_bin ][j_bin ].numpy()[0] == 0.0:
                            continue

                        gradient = grad(bin_contents[i_bin][j_bin], osc_par)
                        fisher_info = tensor.pow(gradient, 2.0)

                    if binning.n_dims == 1:
                        gradients[osc_par_name][i_bin - 1] = gradient.numpy()[0]
                        fisher_informations[osc_par_name][i_bin - 1] = fisher_info.numpy()[0]
                    elif binning.n_dims == 2:
                        gradients[osc_par_name][i_bin - 1][j_bin -1] = gradient.numpy()[0]
                        fisher_informations[osc_par_name][i_bin - 1][j_bin - 1] = fisher_info.numpy()[0]

        if self.make_plots:
            for osc_par in self.oscillator.parameters.keys():

                for data_dict, label in zip([gradients], ["Gradient"]):
                    fig, ax = plt.subplots()
                    
                    data = data_dict[osc_par]

                    data[data == 0] = np.nan

                    if binning.n_dims == 2:
                        u_bins, v_bins = binning.bins

                        mappable = ax.pcolormesh(binning.bins[0], binning.bins[1], data.T)

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

        return fisher_informations