import typing
from enum import IntEnum
import pickle

import uproot
from matplotlib import pyplot as plt
import numpy as np

from nuTens.tensor import tensor, Tensor
from nuTens.autograd import grad

from nuPhase.utils import Molecule
from nuPhase.oscillator import OscillationCalculator
from nuPhase.event import Event, Particle
from nuPhase.selection import SelectionBase

from tqdm import tqdm

class NuFlavour(IntEnum):

    electron = 0
    muon = 1
    tau = 2

class Binning:
    """Represents binning for use in analyses
    """

    def __init__(
        self,
        variables: typing.Tuple[str],
        n_bins: typing.Tuple[int] = None,
        ranges: typing.Tuple[typing.Tuple[float]] = None,
        bins: typing.List[np.array] = None,
    ):
        
        self.variables = variables
        self.n_dims = len(variables)

        if bins is None:
            assert len(variables) == len(n_bins) == len(ranges), f"Bad binning! lenght of variables ({len(variables)}) must be equal to length of n_bins ({len(n_bins)} and ranges ({len(ranges)})!!!"

            self.n_bins = n_bins
            self.ranges = ranges

            self.bins = []
            for var, n, range in zip(variables, n_bins, ranges):

                assert len(range) == 2, f"bad range for var {var}, must be (low, up)"

                self.bins.append(np.linspace(range[0], range[1], n + 1))

        else:

            assert len(bins) == self.n_dims, f"bad bins! must have same number of dimensions as number of variables!! was {len(bins)} vs {self.n_dims}"
            self.bins = bins
            self.n_bins = [b.shape[0] - 1 for b in bins]
            self.ranges = [(b[0], b[-1]) for b in bins]

    def __eq__(self, other):

        if type(other) is not Binning:
            return False
        
        if self.n_dims != other.n_dims:
            return False
        
        for myvar, othervar in zip(self.variables, other.variables):
            if myvar != othervar:
                return False
            
        for mybins, otherbins in zip(self.bins, other.bins):
            if not np.all(mybins == otherbins):
                return False
            
        return True

    def digitize(self, values: typing.Union[typing.List[float], float]) -> typing.List[int]:

        _values = values
        n_values = None

        if type(_values) in [list, tuple]:
            assert len(_values) == self.n_dims
            n_values = len(_values)
        elif type(_values) == np.array:
            assert len(_values.shape) == 1
            n_values = self.n_dims
        elif type(_values) == float:
            assert self.n_dims == 1
            _values = [_values]
            n_values = 1

        bin_indices = []

        for i_val in range(n_values):

            bin_indices.append(np.digitize(_values[i_val], self.bins[i_val]))

        return bin_indices

    def get_bin_edges(self, variable: str = None):

        if variable is None:
            return self.bins

        else:
            i_var = self.variables.index(variable)
            return self.bins[i_var]

    def get_n_bins(self, variable: str = None):

        if variable is None:
            return self.n_bins

        else:
            i_var = self.variables.index(variable)
            return self.n_bins[i_var]
        
    def get_range(self, variable: str = None):

        if variable is None:
            return self.ranges

        else:
            i_var = self.variables.index(variable)
            return self.ranges[i_var]

class Parameters:

    def __init__(
            self,
            pot:float,
            target_material: Molecule,
            target_mass: float,
        ):
        
        self.pot: float = pot
        self.target_material: Molecule = target_material
        self.target_mass: float = target_mass


class NuisanceFile: 
    """Little convenience class for accessing data in nuisance files
    """

    def __init__(
        self,
        file_name: str,
        pre_selection: str = None
    ):

        self.pre_selection = pre_selection

        with uproot.open(file_name) as file:

            self._data = file["FlatTree_VARS"]
            assert self._data is not None, f"No FlatTree_VARS tree in input file {file_name}! is this really a nuisance flattree???"
            
            self.num_entries = self._data.num_entries

            self.flux_hist = file["FlatTree_FLUX"]
            self.scale_factor = self.get_array("fScaleFactor")[0]

    def __getitem__(self, key: str):
        
        return self._data[key]
    
    def get_arrays(self, keys: typing.List[str]):

        return self._data.arrays(keys, self.pre_selection, library="np")
    
    def get_array(self, key: str):

        return self._data.arrays(key, self.pre_selection, library="np")[key]
    
    def keys(self): 

        return self._data.keys()
    

class SubSample:

    def __init__(
        self, 
        label: str, 
        target_material: Molecule, 
        initial_flavour: NuFlavour, 
        final_flavour: NuFlavour,
        oscillator: OscillationCalculator = None, 
        base_pot=1e21
    ):

        self.label: str                        = label
        self.base_pot:float                    = base_pot
        self.target_material: Molecule         = target_material

        self.initial_flavour: NuFlavour        = initial_flavour
        self.final_flavour: NuFlavour          = final_flavour
        self.oscillator: OscillationCalculator = oscillator

        ## these should be filled later
        self.events: typing.List[Event] = []
        self._flux_hist: np.array       = None

        ## The weight that should be applied to events in this sample to 
        ## recover the cross section that was used to generate the events
        self.fixed_xsec_weight = None

        ## The integrated flux for this subsample
        self.integrated_flux = None

    def _get_event_info(self, file: NuisanceFile, aux_vars: typing.List[str], progress_bar: bool, max_n_events: int = None) -> None:
        """read event info from input file and turn it into an array of events
        """

        ## read arrays of particle info
        n_particle_array = file.get_array("nfsp")
        px               = file.get_array("px")
        py               = file.get_array("py")
        pz               = file.get_array("pz")
        energies         = file.get_array("E")
        pdg              = file.get_array("pdg")

        ## read event level variables
        nu_pdg      = file.get_array("PDGnu")
        nu_energies = file.get_array("Enu_true")
        modes       = file.get_array("Mode")

        ## read auxilary variables specified by user
        aux_var_arrays = [
            file.get_array(aux_var) for aux_var in aux_vars
        ]

        n_events_to_read = n_particle_array.shape[0]
        if max_n_events is not None:
            n_events_to_read = min(max_n_events, n_events_to_read)

        iterable = range(n_events_to_read)
        if progress_bar:
            iterable = tqdm(range(n_events_to_read), desc = f"Reading events for subsample {self.label}")

        for i_event in iterable:

            event = Event(e_nu = nu_energies[i_event], mode = modes[i_event], nu_pdg = nu_pdg[i_event])

            event.add_particles_from_arrays(px=px[i_event], py=py[i_event], pz=pz[i_event], energies=energies[i_event], pdg=pdg[i_event])

            ## add auxilary variables
            for aux_var, array in zip(aux_vars, aux_var_arrays):
                event.aux_vars[aux_var] = array[i_event]
            
            self.events.append(event)

    def shallow_copy(self) -> 'SubSample':
        """Makes a very shallow copy of this SubSample with all the same member variable values but an empty event list

        :return: copy
        :rtype: SubSample
        """
         
        new_subsample = SubSample(
            label = self.label, 
            target_material = self.target_material, 
            initial_flavour = self.initial_flavour, 
            final_flavour = self.final_flavour, 
            base_pot = self.base_pot
        )

        new_subsample.flux_hist         = self.flux_hist
        new_subsample.integrated_flux   = self.integrated_flux
        new_subsample.fixed_xsec_weight = self.fixed_xsec_weight
        new_subsample.oscillator        = self.oscillator

        return new_subsample
    
    def fill_from_file(self, file: NuisanceFile, auxilary_variables = ["Q2", "q0", "q3"], progress_bar: bool = False, max_n_events: int = None) -> 'SubSample':
        """Fill this subsample with events read in from a nuisance flat tree
        """

        self._get_event_info(file, auxilary_variables, progress_bar=progress_bar, max_n_events=max_n_events)

        self.flux_hist = file.flux_hist.to_numpy()
        
        self.fixed_xsec_weight = file.scale_factor
        self.integrated_flux   = self.get_integrated_flux()

        return self


    def get_integrated_flux(self, bin_width_normalised=True) -> float:

        assert self.flux_hist is not None, "hmmmm, flux hist is None. Has this subsample been initialised properly????"

        counts, bin_edges = self.flux_hist ## counts are in units of [1 / (cm^2 * 50 MeV * 10^21 POT)]
        bin_widths = ( bin_edges[1:] - bin_edges[:-1] ) / 0.05 ## bin widths "in units of [50MeV]"

        ret = None

        if bin_width_normalised:
            ret = (counts * bin_widths).sum() ## flux in units of [1 / (cm^2 * 10^21 POT)]

        else:
            ret = counts.sum()

        return ret
    
    def get_xsec_weight(self) -> float:

        return self.fixed_xsec_weight
    
    def get_pot_weight(self, pot: float) -> float:

        return pot / self.base_pot

    def get_event_scaling(self, target_mass: float, pot: float) -> float:
        """Get the scaling that should be applied to events in this sub-sample to estimate event rates assuming the given target mass and POT
        """

        n_nucleons = self.target_material.get_n_nucleons(target_mass)
        pot_weight = self.get_pot_weight(pot)
        
        return self.integrated_flux * self.fixed_xsec_weight * n_nucleons * pot_weight

    def get_array(self, key: str, cut: typing.Callable = None) -> np.array:
        """Get an array of event level variables for each event in this SubSample
        
        returns an array containing values for each event filled with the specified variable.
        Can specify a cut which should be a function that takes an event as input and returns true or false.
        """

        values = []
        for event in self.events:

            if cut is None or cut(event):
                values.append(event.get_var(key))

        return np.array(values)
    
    def apply_selection(self, selection: SelectionBase, progress_bar: bool = False) -> 'SubSample':
        """Apply a selection to the events in this subsample
        
        Will return a copy of this subsapmple with only events that pass the selection in it
        """

        new_subsample = self.shallow_copy()

        iterator = self.events
        if progress_bar:
            iterator = tqdm(self.events, desc = f"applying [{selection.name}] to {self.label}")
        
        for event in iterator:

            if selection.apply(event):

                new_subsample.events.append(event)

        return new_subsample
    
    def oscillate_events(self, progress_bar: bool = False, save_gradients: bool = False, second_deriv: bool = False) -> None:
        """Calculate oscillations for each event and fill auxilary variable "osc_weight" with tensor containing oscillation weight

        If there is no oscillator for this subsample then the oscillation weight will just be 1

        TODO: add option to also save the gradients for each osc parameter, maybe also second derivatives, fisher info etc.
        """

        if self.oscillator is None:

            for event in self.events:
                event.aux_vars["osc_weight"] = tensor.ones([1])

            return
        
        energies = self.get_array("Enu_true")
        osc_probs = self.oscillator.calculate_osc_probs(energies)
        osc_weights = osc_probs.get_values(["...", self.initial_flavour, self.final_flavour])

        iterator = self.events
        if progress_bar:
            iterator = tqdm(self.events, desc="oscillatin' events")
        
        for i_event, event in enumerate(iterator):

            event_weight = osc_weights.get_values([i_event])

            event.aux_vars["osc_weight"] = event_weight

            if save_gradients:

                for par_name, parameter in zip(self.oscillator.parameters.keys(), self.oscillator.parameters.values()):

                    parameter_grad_tensor = grad(event_weight, parameter)
                    parameter_grad = parameter_grad_tensor.numpy()[0]

                    event.aux_vars[f"osc_weight_{par_name}_grad"] = parameter_grad

                    if second_deriv:

                        parameter_second_deriv = grad(parameter_grad_tensor, parameter)

                        event.aux_vars[f"osc_weight_{par_name}_second_grad"] = parameter_second_deriv

                self.oscillator.zero_grad()

    def get_event_rate(self, binning: Binning, target_mass: float, pot: float, cut: typing.Callable = None, weight_var: str = None):

        data_list = []

        ## keep track of num of events passing the cut
        n_events = 0

        for iVar in range(binning.n_dims):
            array = self.get_array(binning.variables[iVar], cut)

            data_list.append(array)

            if iVar == 0:
                n_events = array.shape[0]

        ## caclulate oscillation weights if needed
        osc_weights = np.ones(n_events)
        if self.oscillator is not None:
            energies = self.get_array("Enu_true", cut)
            osc_probs = self.oscillator.calculate_osc_probs(energies)
            osc_weights = osc_probs.numpy()[:, self.initial_flavour, self.final_flavour]

        ## if weight variable specified make weight array
        weight_array = None
        if weight_var is not None:
            weight_array = self.get_array(weight_var, cut=cut)
        else:
            weight_array = np.ones(n_events)

        ## now make the histogram
        hist, _ = np.histogramdd(data_list, bins = binning.bins, weights = osc_weights * weight_array)

        return hist * self.get_event_scaling(target_mass, pot)

class Sample:

    def __init__(
        self,
        binning: Binning,
        subsamples: typing.List[SubSample],
        parameters: Parameters,
        name: str
    ):

        self.name = name
        self.n_dims = binning.n_dims
        self.binning = binning
        self.subsamples = subsamples
        self.parameters = parameters
        
        self.events: typing.List[Event] = []
        for subsample in self.subsamples:
            self.events += subsample.events

    def oscillate_events(self, progress_bar: bool = False, save_gradients: bool = False) -> None:
        """Calculate oscillations for each subsample

        Just calls SubSample.oscillate_events() on each subsample
        """

        for subsample in self.subsamples:
            subsample.oscillate_events(progress_bar=progress_bar, save_gradients=save_gradients)

    def apply_selection(self, selection: SelectionBase, progress_bar: bool = False) -> 'Sample':

        new_subsamples = []
        for subsample in self.subsamples:

            new_subsample = subsample.apply_selection(selection=selection, progress_bar=progress_bar)
            new_subsamples.append(new_subsample)

        new_sample = Sample(
            binning = self.binning,
            subsamples = new_subsamples,
            parameters = self.parameters,
            name = f'{self.name} [{selection.name}]'
        )

        return new_sample

    
    def imshow(self, axis, data_override: np.array, binning: Binning = None, z_label: str = None, *imshow_args):
        
        if binning is None:
            binning = self.binning

        assert binning.n_dims == 2, "need 2 dims for imshowing!!!"

        u_bins, v_bins = binning.bins

        dat = data_override

        mappable = axis.imshow(dat.T, extent=(u_bins[0], u_bins[-1], v_bins[0], v_bins[-1]), origin="lower", *imshow_args)

        cbar = plt.colorbar(mappable)
        if z_label is None:
            cbar.set_label(f"N Events / {self.parameters.pot:.2E} POT / {self.parameters.target_mass:.2E} kg")
        else:
            cbar.set_label(z_label)
                
        axis.set_title(f"{self.name}")

        plt.xlabel(self.binning.variables[0])
        plt.ylabel(self.binning.variables[1])

    def get_event_rates(
            self,
            binning: Binning = None,
            keep_zero = True,
            cut: typing.Callable = None,
            weight_var: str = None
        ):

        if binning is None:
            binning = self.binning

        hist_total = np.zeros(binning.n_bins)

        for subsample in self.subsamples:

            hist_total += subsample.get_event_rate(
                binning, 
                target_mass=self.parameters.target_mass, 
                pot = self.parameters.pot, 
                cut = cut,
                weight_var = weight_var
            )

        if not keep_zero:
            hist_total[hist_total == 0] = np.nan

        return hist_total
    
    def get_array(self, key: str, cut: typing.Callable = None) -> np.array:
        """Get an array of event level variables for each event in this SubSample
        
        returns an array containing values for each event filled with the specified variable.
        Can specify a cut which should be a function that takes an event as input and returns true or false.
        """

        values = []
        for event in self.events:

            if cut is None or cut(event):
                values.append(event.get_var(key))

        return np.array(values)

    def to_file(self, file_name:str) -> None:

        with open(file_name, "wb") as file:

            pickler = pickle.Pickler(file)
            pickler.dump(self)

    @staticmethod
    def from_file(file_name: str) -> 'Sample':

        with open(file_name, "rb") as file:

            unpickler = pickle.Unpickler(file)
            return unpickler.load()