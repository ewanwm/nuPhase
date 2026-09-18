import typing

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from nuPhase.sample import Sample, Binning
from nuPhase.oscillator import OscillationCalculator
from nuPhase.event import Event
from nuPhase.modules.base import TransformationBase

class CalculateFisherInfo(TransformationBase):
    """Calculates the fisher information of a sample with respect to the available oscillation parameters

    Will calculate the event-by event fisher information and add the variables <PARAMETER_NAME>_fisher_info to
    each event
    """

    def __init__(
        self,
        oscillator: OscillationCalculator,
        make_plots: bool = True,
        plot_file_name: str = "Fisher-info.pdf",
    ):

        self.make_plots = make_plots
        self.oscillator = oscillator
        self.plot_file_name = plot_file_name

        self.fisher_info = None
        self.event_rates = None
        self.per_event_fisher_info = None

    def _initialise(self, sample: Sample) -> None:

        if self.make_plots:

            self._pdf = PdfPages(self.plot_file_name)

        ## get necessary global sample quantities
        self.fisher_info = self._get_fisher_info(sample=sample, binning=sample.binning)
        self.event_rates = sample.get_event_rates(binning=sample.binning, keep_zero=True)

        ## get event by event fisher info
        self.per_event_fisher_info = {}
        for parameter_name in self.oscillator.parameters.keys():

            self.per_event_fisher_info[parameter_name] = self.fisher_info[parameter_name] / self.event_rates

    def _finalise(self):

        if self.make_plots:

            for parameter_name in self.oscillator.parameters.keys():

                if self.sample.binning.n_dims == 1:

                    fig, ax = plt.subplots()

                    dat = self.fisher_info[parameter_name][...]
                    dat[dat == 0.0] = np.nan

                    plt.stairs(dat, self.sample.binning.bin_edges[0])
                    plt.xlabel(f"{self.sample.binning.variables[0]}")
                    plt.ylabel("Fisher info")
                    plt.title(f"{self.sample.name} {parameter_name} \nFisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)
                    fig, ax = plt.subplots()

                    dat = self.per_event_fisher_info[...]
                    dat[dat == 0.0] = np.nan

                    plt.stairs(dat, self.sample.binning.bin_edges[0])
                    plt.xlabel(f"{self.sample.binning.variables[0]}")
                    plt.ylabel("Fisher info")
                    plt.title(
                        f"{self.sample.name} {parameter_name} \nPer-event Fisher Information"
                    )
                    self._pdf.savefig(fig)

                    plt.close(fig)

                elif self.sample.binning.n_dims == 2:
                    fig, ax = plt.subplots()

                    dat = self.fisher_info[parameter_name][...]
                    dat[dat == 0.0] = np.nan

                    self.sample.imshow(ax, data_override=dat, z_label="Fisher Information")
                    plt.title(f"{self.sample.name} {parameter_name} \nFisher Information")
                    self._pdf.savefig(fig)

                    plt.close(fig)
                    fig, ax = plt.subplots()

                    dat = self.per_event_fisher_info[...]
                    dat[dat == 0.0] = np.nan

                    self.sample.imshow(
                        ax, data_override=dat, z_label="Per-event Fisher Information"
                    )
                    plt.title(
                        f"{self.sample.name} {parameter_name} \nPer-event Fisher Information"
                    )
                    self._pdf.savefig(fig)

                    plt.close(fig)

                else:
                    raise ValueError(
                        "Can't make fisher info plots for n-dims != 1 or 2 :("
                    )

        self._pdf.close()

    def _apply(self, event: Event):

        for parameter_name in self.oscillator.parameters.keys():

            bins = tuple(
                self.sample.binning.digitize(
                    [event.get_var(var) for var in self.sample.binning.variables]
                )
            )

            try:
                event.aux_vars[f"{parameter_name}_fisher_info"] = (
                    self.per_event_fisher_info[bins]
                )
            except IndexError:
                event.aux_vars[f"{parameter_name}_fisher_info"] = 0.0
                continue

    def _get_fisher_info(
        self, sample: Sample, binning: Binning = None
    ) -> typing.Dict[str, np.ndarray]:
        """Calculates fisher information for the sample

        :param sample: The sample
        :type sample: Sample
        :param binning: Binning to calculate fisher info in, if None then will use the binning of the sample, defaults to None
        :type binning: Binning, optional
        :return: Dictionary whose keys are names of oscillation parameters and values ar the bin-by-bin fisher information
        :rtype: typing.Dict[str, np.ndarray]
        """

        if binning is None:
            binning = sample.binning

        ## calculate the osc probs
        sample.oscillate_events(progress_bar=True, save_gradients=True)

        if self.make_plots:
            for subsample in sample.subsamples:

                fig = plt.figure()
                fig.clear()

                plt.stairs(subsample.binned_osc_probs, subsample.osc_energy_binning)
                plt.title(f"{subsample.name} Binned Oscillations")
                plt.xlabel("Enu [GeV]")

                self._pdf.savefig(fig)

                for par_name in self.oscillator.parameters.keys():

                    fig.clear()
                    plt.stairs(
                        subsample.binned_gradients[par_name],
                        subsample.osc_energy_binning,
                    )
                    plt.title(f"{subsample.name} Binned {par_name} Gradients")
                    plt.xlabel("Enu [GeV]")

                    self._pdf.savefig(fig)

        gradients = {}
        fisher_informations = {}

        for osc_par in self.oscillator.parameters.keys():

            gradients[osc_par] = sample.get_event_rates(
                weight_var=f"osc_weight_{osc_par}_grad"
            )

            fisher_informations[osc_par] = gradients[osc_par] * gradients[osc_par]

        if self.make_plots:
            for osc_par in self.oscillator.parameters.keys():

                for data_dict, label in zip([gradients], ["Gradient"]):
                    fig, ax = plt.subplots()

                    data = data_dict[osc_par]

                    data[data == 0] = np.nan

                    if binning.n_dims == 2:

                        mappable = ax.pcolormesh(
                            binning.bin_edges[0], binning.bin_edges[1], data.T
                        )

                        ax.set_xlabel(binning.variables[0])
                        ax.set_ylabel(binning.variables[1])

                        cbar = plt.colorbar(mappable)
                        cbar.set_label(f"{label}")

                    elif binning.n_dims == 1:

                        ax.stairs(data, binning.bin_edges[0])
                        ax.set_xlabel(binning.variables[0])
                        ax.set_ylabel(f"{label}")

                    plt.title(f"{sample.name} {osc_par} {label}")

                    self._pdf.savefig(fig)

                    plt.close(fig)

        return fisher_informations


class ApplyVariableSmearing(TransformationBase):
    """Smear a truth variable to mimic finite detector resolution
    """

    def __init__(
        self, true_var: str, smeared_var: str, smear_function: typing.Callable
    ):
        self.true_var = true_var
        self.smeared_var = smeared_var
        self.smear_function = smear_function

        self.generator = np.random.default_rng(seed=None)

    def _apply(self, event: Event):

        true_var = event.get_var(self.true_var)

        scale = self.smear_function(true_var)

        event.aux_vars[self.smeared_var] = self.generator.normal(
            loc=true_var, scale=scale
        )
