import typing
from enum import IntEnum

import uproot
from matplotlib import pyplot as plt
import numpy as np

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

        ## The weight that should be applied to events in this sample to 
        ## recover the flux that was used to generate the events
        self.fixed_flux_weight = None

    def _get_event_info(self, file: NuisanceFile, aux_vars: typing.List[str], progress_bar: bool) -> None:
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

        iterable = range(n_particle_array.shape[0])
        if progress_bar:
            iterable = tqdm(range(n_particle_array.shape[0]), desc = f"Reading events for subsample {self.label}")

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
        new_subsample.fixed_flux_weight = self.fixed_flux_weight
        new_subsample.fixed_xsec_weight = self.fixed_xsec_weight
        new_subsample.oscillator        = self.oscillator

        return new_subsample
    
    def fill_from_file(self, file: NuisanceFile, auxilary_variables = ["Q2", "q0", "q3"], progress_bar: bool = False) -> 'SubSample':
        """Fill this subsample with events read in from a nuisance flat tree
        """

        self._get_event_info(file, auxilary_variables, progress_bar=progress_bar)

        self.flux_hist = file.flux_hist.to_numpy()
        
        self.fixed_xsec_weight = file.scale_factor * file.num_entries
        self.fixed_flux_weight = self.get_integrated_flux() / file.num_entries

        return self


    def get_integrated_flux(self, bin_width_normalised=True) -> float:

        assert self.flux_hist is not None, "hmmmm, flux hist is None. Has this subsample been initialised properly????"

        counts, bin_edges = self.flux_hist
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        ret = None

        if bin_width_normalised:
            ret = (counts * bin_widths / 0.005).sum()

        else:
            ret = counts.sum()

        return ret
    
    def get_xsec_weight(self) -> float:

        return self.fixed_xsec_weight
    
    def get_pot_weight(self, pot: float) -> float:

        return pot / self.base_pot
    
    def get_flux_weight(self):
        
        return self.fixed_flux_weight

    def get_event_scaling(self, target_mass: float, pot: float) -> float:
        """Get the scaling that should be applied to events in this sub-sample to estimate event rates assuming the given target mass and POT
        """

        n_nucleons = self.target_material.get_n_nucleons(target_mass)
        pot_weight = self.get_pot_weight(pot)
        
        return self.fixed_flux_weight * self.fixed_xsec_weight * n_nucleons * pot_weight

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
            iterator = tqdm(self.events, desc = f"applying {selection.name} to {self.label}")
        
        for event in iterator:

            if selection.apply(event):

                new_subsample.events.append(event)

        return new_subsample
    
    def get_event_rate(self, binning: Binning, target_mass: float, pot: float, cut: typing.Callable = None):

        v1 = self.get_array(binning.variables[0], cut)

        v2 = None
        if len(binning.variables) == 2:
            v2 = self.get_array(binning.variables[1], cut)

        ## caclulate oscillation weights if needed
        osc_weights = np.ones(v1.shape)
        if self.oscillator is not None:
            energies = self.get_array("Enu_true", cut)
            osc_probs = self.oscillator.calculate_osc_probs(energies)
            osc_weights = osc_probs.numpy()[:, self.initial_flavour, self.final_flavour]

        ## now make the histogram
        hist = None
        if binning.n_dims == 1:
            hist, _ = np.histogram(v1, bins=binning.bins[0], weights=osc_weights)

        elif binning.n_dims == 2:
            v2 = self.get_array(binning.variables[1], cut)
            hist, _, _ = np.histogram2d(v1, v2, bins=binning.bins, weights=osc_weights)

        else:
            raise NotImplementedError('can only do 1 or 2 variables for now :(')

        return hist * self.get_event_scaling(target_mass, pot)

class Sample:

    def __init__(
        self,
        binning: Binning,
        subsamples: typing.List[SubSample],
        parameters: Parameters,
        name: str
    ):
        
        assert binning.n_dims <= 2, "only support 2d samples!!"

        self.name = name
        self.n_dins = binning.n_dims
        self.binning = binning
        self.subsamples = subsamples
        self.parameters = parameters
        
        self.events = []
        for subsample in self.subsamples:
            self.events += subsample.events

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

    
    def imshow(self, axis, data_override: np.array, *imshow_args):
        
        assert self.n_dins == 2, "need 2 dims for imshowing!!!"

        u_bins, v_bins = self.binning.bins

        dat = data_override

        mappable = axis.imshow(dat.T, extent=(u_bins[0], u_bins[-1], v_bins[0], v_bins[-1]), origin="lower", *imshow_args)

        cbar = plt.colorbar(mappable)
        cbar.set_label(f"N Events / {self.parameters.pot:.2E} POT / {self.parameters.target_mass:.2E} kg")
        
        axis.set_title(f"{self.name}")

        plt.xlabel(self.binning.variables[0])
        plt.ylabel(self.binning.variables[1])

    def get_event_rates(
            self,
            binning: Binning = None,
            keep_zero = True,
            cut: typing.Callable = None
        ):

        if binning is None:
            binning = self.binning

        hist_total = np.zeros(binning.n_bins)

        for subsample in self.subsamples:

            hist_total += subsample.get_event_rate(
                binning, 
                target_mass=self.parameters.target_mass, 
                pot = self.parameters.pot, 
                cut=cut
            )

        if not keep_zero:
            hist_total[hist_total == 0] = np.nan

        return hist_total
    