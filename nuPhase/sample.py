"""This module handles samples and subsamples of events

These provide a handle that can be passed to later downstream 
modules to perform analysis and can also be transformed by 
transformation and selection modules.
"""

import typing
from enum import IntEnum
import pickle
from collections.abc import Iterable
import abc
import copy

import uproot
from matplotlib import pyplot as plt
import numpy as np

from nuTens import tensor
from nuTens.tensor import Tensor
from nuTens.autograd import grad

from tqdm import tqdm

import json
import jsonschema

from nuPhase.materials import Molecule
from nuPhase.oscillator import OscillationCalculator
from nuPhase.event import Event

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nuPhase.modules.base import TransformationBase, SelectionBase


class NuFlavour(IntEnum):
    """Neutrino flavours
    """

    electron = 0
    muon = 1
    tau = 2


def flavour_from_name(name: str) -> NuFlavour:
    """Get neutrino flavour enum from a name string

    :param name: Neutrino flavour name (one of ["nue", "numu", "nutau"])
    :type name: str
    :raises ValueError: if an unknown flavour name is given
    :return: neutrino flavour enum value
    :rtype: NuFlavour
    """

    if name == "nue":
        return NuFlavour.electron
    elif name == "numu":
        return NuFlavour.muon
    elif name == "nutau":
        return NuFlavour.tau
    else:
        raise ValueError(f"Unknown neutrino flavour name specified: {name}")

class Binning:
    """Represents binning for use in analyses"""

    @staticmethod
    def from_file(file_name: str) -> 'Binning':
        """Create a binning object from a json config file

        :param file_name: The name of the config file to read from 
        :type file_name: str
        :return: Binning object with variables and binning defined by the given config file
        :rtype: Binning
        """

        with open(file_name) as json_file:

            ## reat the config file
            dat = json.load(json_file)

        ## create schema for binning config
        schema = {
            "type": "object",
            "properties": {
                "binning": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "proparties": {
                            "variable": {"type": "string"},
                            "bins": {
                                "type": "array"
                            }
                        },
                        "required": ["variable", "bins"]
                    }
                }
            }
        }

        ## validate the user provided json
        jsonschema.validate(instance=dat, schema=schema)

        ## to be filled from file 
        variables = []
        bins      = []

        binning = dat["binning"]
        for variable_binning in binning:
            variables.append(variable_binning["variable"])
            bins.append(np.array(variable_binning["bins"]))

        ## create the binning object
        return Binning(variables=variables, bin_edges=bins)

    def __init__(
        self,
        variables: typing.Tuple[str],
        n_bins: typing.Tuple[int] = None,
        ranges: typing.Tuple[typing.Tuple[float]] = None,
        bin_edges: typing.List[np.array] = None,
    ):
        """Create a new binning

        Can provide *either* n_bins and ranges - then the binning will be `n_bins` evenly spaced bins between lower and upper limits defined by `ranges`

        *or*

        bin_edges, then the binning will be defined by those

        :param variables: Names of variables in the binning
        :type variables: typing.Tuple[str]
        :param n_bins: number of bins, should provide one entry for each variable, defaults to None
        :type n_bins: typing.Tuple[int], optional
        :param ranges: Ranges of binning, should provide one tuple for each entry representing (min_value, max_value), defaults to None
        :type ranges: typing.Tuple[typing.Tuple[float]], optional
        :param bin_edges: _description_, defaults to None
        :type bin_edges: typing.List[np.array], optional
        """

        ## check user has provided valid options
        if bin_edges is None:

            if n_bins is not None or ranges is not None:

                raise ValueError("Should provide *either* bin_eges, or n_bins and ranges")

        self.variables: typing.List[str] = variables
        self.n_dims: int = len(variables)

        self.bin_edges: typing.List[np.array] = None
        self.n_bins: typing.List[int] = None
        self.ranges: typing.Tuple[typing.Tuple[float]] = None

        if bin_edges is None:
            if not (
                len(variables) == len(n_bins) == len(ranges)
            ):
                raise ValueError(f"Bad binning! lenght of variables ({len(variables)}) must be equal to length of n_bins ({len(n_bins)} and ranges ({len(ranges)})!!!")

            self.n_bins = n_bins
            self.ranges = ranges

            self.bin_edges = []
            for var, n, range in zip(variables, n_bins, ranges):

                assert len(range) == 2, f"bad range for var {var}, must be (low, up)"

                self.bin_edges.append(np.linspace(range[0], range[1], n + 1))

        else:

            if not (
                len(bin_edges) == self.n_dims
            ):
                raise ValueError(f"bad bins! must have same number of dimensions as number of variables!! was {len(bin_edges)} vs {self.n_dims}")

            self.bin_edges = bin_edges
            self.n_bins = [b.shape[0] - 1 for b in bin_edges]
            self.ranges = [(b[0], b[-1]) for b in bin_edges]

    def __eq__(self, other):

        if type(other) is not Binning:
            return False

        if self.n_dims != other.n_dims:
            return False

        for myvar, othervar in zip(self.variables, other.variables):
            if myvar != othervar:
                return False

        for mybins, otherbins in zip(self.bin_edges, other.bin_edges):
            if not np.all(mybins == otherbins):
                return False

        return True

    def digitize(
        self, values: typing.Union[typing.List[float], float]
    ) -> typing.List[int]:
        """Find which bins some data point falls into

        :param values: The coordinates of the data point. Should be an array with one entry for each dimension of the binning
        :type values: typing.Union[typing.List[float], float]
        :return: The indices of the bins the provided values fall into, one entry for each dimension
        :rtype: typing.List[int]
        """

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

            bin_indices.append(np.digitize(_values[i_val], self.bin_edges[i_val]))

        return bin_indices

    def get_bin_edges(self, variable: str = None) -> typing.List[np.array]:
        """
        :param variable: If provided, will return the bin edges for one specific variable - otherwise returns list of bin edges for all variables, defaults to None
        :type variable: str, optional
        :raises ValueError: If the specified variable does not exist in this binning
        :return: The bin edges
        :rtype: typing.List[np.array]
        """

        if variable is None:
            return self.bin_edges

        else:
            if variable not in self.variables:
                raise ValueError(f"variable {variable} does not exist in this binning. (have {self.variables})")
            
            i_var = self.variables.index(variable)
            return self.bin_edges[i_var]

    def get_n_bins(self, variable: str = None) -> typing.List[int]:
        """
        :param variable: If provided, will return the number of bins for one specific variable - otherwise returns list of number of bins for all variables, defaults to None
        :type variable: str, optional
        :raises ValueError: If the specified variable does not exist in this binning
        :return: The number of bins
        :rtype: typing.List[int]
        """

        if variable is None:
            return self.n_bins

        else:
            if variable not in self.variables:
                raise ValueError(f"variable {variable} does not exist in this binning. (have {self.variables})")
            
            i_var = self.variables.index(variable)
            return self.n_bins[i_var]

    def get_range(self, variable: str = None) -> typing.Tuple[typing.Tuple[float]]:
        """
        :param variable: If provided, will return the range for one specific variable - otherwise returns list of ranges for all variables, defaults to None
        :type variable: str, optional
        :raises ValueError: If the specified variable does not exist in this binning
        :return: The number of bins
        :rtype: typing.Tuple[typing.Tuple(float)]
        """

        if variable is None:
            return self.ranges

        else:
            if variable not in self.variables:
                raise ValueError(f"variable {variable} does not exist in this binning. (have {self.variables})")
            
            i_var = self.variables.index(variable)
            return self.ranges[i_var]

    def project(self, variables: typing.Union[str, typing.Iterable[str]]) -> "Binning":
        """Create a projection of this binning onto the specified variables

        :param variables: The variables to project into
        :type variables: typing.union[str, typing.Iterable[str]]
        :raises ValueError: If any of the variables are not contained in this binning
        :return: Projected binning
        :rtype: Binning
        """

        _variables = None

        if type(variables) == str:

            _variables = [variables]

        if isinstance(variables, Iterable):

            _variables = variables

        ## check that all variables are actually in this binning
        for var in _variables:
            if var not in self.variables:
                raise ValueError(f"Variable {var} not found in binning!")

        ret = Binning(_variables, bin_edges=[self.get_bin_edges(var) for var in _variables])

        return ret

    def get_2d_projections(self) -> typing.List["Binning"]:
        """Get all possible 2D projections for this binning
        
        :return: Projected binnings
        :rtype: typing.List["Binning"]
        """

        ret = []

        if self.n_dims < 2:
            return ret

        for iDim in range(self.n_dims):
            for jDim in range(iDim + 1, self.n_dims):

                ret.append(self.project([self.variables[iDim], self.variables[jDim]]))

        return ret

    def get_1d_projections(self) -> typing.List["Binning"]:
        """Get all 1D projections for this binning

        :return: Projected binnings
        :rtype: typing.List["Binning"]
        """

        ret = []

        for iDim in range(self.n_dims):

            ret.append(self.project([self.variables[iDim]]))

        return ret


class SubSampleParameters:
    """Holds sample parameters
    """

    def __init__(
        self,
        pot: float,
        target_material: Molecule,
        target_mass: float,
        initial_flavour: NuFlavour,
        final_flavour: NuFlavour,
        antineutrino: bool,
    ):
        """
        :param pot: Desired POT
        :type pot: float
        :param target_material: Desired target material
        :type target_material: Molecule
        :param target_mass: Desired target mass (in kg)
        :type target_mass: float
        :param initial_flavour: The "initial" (unoscillated) flavour represented by the subsample
        :type initial_flavour: NuFlavour
        :param final_flavour: The "final" (oscillated) flavour represented by the subsample
        :type final_flavour: NuFlavour
        :param antinuetrino: Flag which should be True if the sample is in anti-neutrino mode
        :type antineutrino: bool
        """

        self.pot: float = pot
        self.target_material: Molecule = target_material
        self.target_mass: float = target_mass
        self.initial_flavour: NuFlavour = initial_flavour
        self.final_flavour: NuFlavour = final_flavour
        self.antinu: bool = antineutrino

    def __str__(self):

        ret_str = ""
        ret_str += "SubSamplePArameters:\n"
        for key, value in zip(self.__dict__.keys(), self.__dict__.values()):

            ret_str += f"  - {key}: {value}\n"

        return ret_str

class NuisanceFile:
    """Little convenience class for accessing data in nuisance files
    
    This is really just a wrapper class for accessing uproot objects.
    To access the data stored in a file you should use the `with` keyword like:

    .. code-block:: python

        file = NuisanceFile(...)

        with file as f:
            ## do stuff with data stored in file

        ## blablabla

        This provides a safe way of accessing the data stored in the nuisance file 
        without consuming unnnecessary memory resources.

    """

    def __init__(self, file_name: str, pre_selection: str = None):
        """
        :param file_name: The path to the nuisance file to be read
        :type file_name: str
        :param pre_selection: ROOT style selection to apply when reading the file e.g. (MODE==1) to read only CCQE events - maybe useful to speed up large file reading if you are interested only in a subset of events, defaults to None
        :type pre_selection: str, optional
        """

        self.pre_selection: str = pre_selection

        self.file_name = file_name

        self.file = None

        self._data = None
        self.num_entries = None
        self.flux_hist = None
        self.scale_factor = None

    def __enter__(self):

        self.file = uproot.open(self.file_name)

        self._data = self.file["FlatTree_VARS"]
        assert (
            self._data is not None
        ), f"No FlatTree_VARS tree in input file {self.file_name}! is this really a nuisance flattree???"

        self.num_entries = self._data.num_entries

        self.flux_hist = self.file["FlatTree_FLUX"]
        self.scale_factor = self.get_array("fScaleFactor")[0]

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):

        self.file.close()
        self.file = None

        self._data = None
        self.num_entries = None
        self.flux_hist = None
        self.scale_factor = None

    def __getitem__(self, key: str):

        return self._data[key]

    def get_arrays(self, keys: typing.List[str]) -> np.ndarray:
        """Read array of branch values from the file

        :param keys: The names of the branches to be read
        :type keys: typing.List[str]
        :return: array of values - one "row" for each file entry
        :rtype: np.ndarray
        """

        return self._data.arrays(keys, self.pre_selection, library="np")

    def get_array(self, key: str) -> np.ndarray:
        """Read array of values for a single branch from the file

        :param key: The names of the branch to be read
        :type key: str
        :return: array of values - one "row" for each file entry
        :rtype: np.ndarray
        """

        return self._data.arrays(key, self.pre_selection, library="np")[key]

    def keys(self) -> typing.List[str]:
        """Get the available keys in this file

        :return: available keys
        :rtype: typing.List[str]
        """

        return self._data.keys()


class SampleBase(abc.ABC):
    """The base class for Sample and SubSample objects
    """

    def __init__(self, name: str):

        ## name of the sample
        self.name: str = name

        ## events within this sample
        self.events: typing.List[Event] = []
        
    def get_array(self, key: str, cut: typing.Callable = None) -> np.array:
        """Get an array of event level variables for each event in this sample

        returns an array containing values for each event filled with the specified variable.
        Can specify a cut which should be a function that takes an event as input and returns true or false.
        """

        values = []
        for event in self.events:

            if cut is None or cut(event):
                values.append(event.get_var(key))

        return np.array(values, dtype=float)

    def apply_selection(
        self, selection: 'SelectionBase', progress_bar: bool = False, strip_particle_info: bool = False
    ) -> "SampleBase":
        """Apply a selection to the events in this sample

        Will return a copy of this subsapmple with only events that pass the selection in it

        .. warning::
            
            strip_particle_info will This will change all references to event so should be used for the last transformation / selection in the chain
         
        """

        ## TODO: Add option to apply in place - don't make a copy and actually alter the original sample object - saving memory

        ## make a shallow copy of this sample
        new_sample = copy.copy(self)
        new_sample.events = []

        iterator = self.events
        if progress_bar:
            iterator = tqdm(
                self.events, desc=f"applying selection [{selection.name}] to {self.name}"
            )

        ## apply the selection
        for event in iterator:

            if selection.apply(event):

                if strip_particle_info:

                    del event.particles

                new_sample.events.append(event)
                

        return new_sample

    def apply_transformation(self, transformation: 'TransformationBase', progress_bar: bool = False, strip_particle_info: bool =False) -> "SampleBase":
        """Apply a transformation to all events in this sample

        .. warning::
            
            strip_particle_info will This will change all references to event so should be used for the last transformation / selection in the chain

        :param transformation: The transformation to be applied
        :type transformation: 'TransformationBase'
        :param progress_bar: If true, will print a progress bar showing how many events have been processed, defaults to False
        :type progress_bar: bool, optional
        :param strip_particle_info: If true, all particle level info will be stripped from the event - this saves memory but means you can't apply any more selections or transformations that require particle level info
        :type strip_particle_info: bool, optional
        :return: This sample (after transformation has been applied)
        :rtype: SampleBase
        """

        ## TODO: Add option to apply in place or make a copy

        iterator = self.events
        if progress_bar:
            iterator = tqdm(
                self.events, desc=f"applying transformation [{type(transformation).__name__}] to {self.name}"
            )

        ## apply the selection
        for event in iterator:

            transformation.apply(event)

            if strip_particle_info:

                del event.particles
            
        return self

    
    def to_file(self, file_name: str, keep_tensors: bool = False) -> None:
        """Dump this object to a file

        :param file_name: path to the file to save the object to
        :type file_name: str
        :param keep_tensors: If True, will save Tensor objects that are stored in events. This means that differentiable quantities are preserved but the file will be a *lot* larger, defaults to False
        :type keep_tensors: bool, optional
        """

        ## strip out tensor objects by default - they really beef up file sizes
        if not keep_tensors:

            for event in self.events:

                to_delete = []

                for var_name, var in event.aux_vars.items():

                    if type(var) == Tensor:

                        to_delete.append(var_name)

                for var_name in to_delete:

                    del event.aux_vars[var_name]

        ## write the file
        with open(file_name, "wb") as file:

            pickler = pickle.Pickler(file)
            pickler.dump(self)

    @staticmethod
    def from_file(file_name: str) -> "SampleBase":
        """Create a Sample or Subsample from a file on disk

        :param file_name: Path to the file
        :type file_name: str
        :return: The recreated sample or subsample
        :rtype: SampleBase
        """

        with open(file_name, "rb") as file:

            unpickler = pickle.Unpickler(file)
            return unpickler.load()


class SubSample(SampleBase):
    """Represents a subsample of events

    Could be e.g. a single oscillation channel
    """

    def __init__(
        self,
        name: str,
        parameters: SubSampleParameters,
        oscillator: OscillationCalculator = None,
        base_pot: float = 1e21,
        do_binned_osc_probs: bool = True,
        osc_energy_binning: np.array = np.linspace(0.0, 5.0, 1000),
    ):
        """Constructor
        
        :param name: A name for the subsample - will be used in plots and printouts
        :type name: str
        :param parameters: describes the parameters for this subsample
        :type parameters: SubSampleParameters
        :param oscillator: If supplied, this will be used to calculate oscillation probabilities, if not, no oscillations will be applied, defaults to None
        :type oscillator: OscillationCalculator, optional
        :param base_pot: The POT that was used to generate the Monte Carlo for this subsample, defaults to 1e21
        :type base_pot: float, optional
        :param do_binned_osc_probs: Whether to use binned oscillation probabilities for this subsample (assuming an oscillation calculator was supplied), defaults to True
        :type do_binned_osc_probs: bool, optional
        :param osc_energy_binning: The binning to use when calculating binned oscillation probabilities, defaults to np.linspace(0.0, 5.0, 1000)
        :type osc_energy_binning: np.array, optional
        """

        ## TODO: Move the binned oscillation stuff to OscillationCalculator?
        ## probably have binned oscillation stuff dealt with in the OscillationCalculator and move oscillation parameters to another class that could be shared between oscillationCalculator objects

        self.name: str = name
        self.base_pot: float = base_pot
        self.parameters: SubSampleParameters = parameters
        self.oscillator: OscillationCalculator = oscillator

        ## these should be filled later
        self.events: typing.List[Event] = []
        self.flux_hist: np.array = None
        self.flux_binning: np.array = None

        ## binned oscillation stuff
        self.do_binned_osc_probs = do_binned_osc_probs
        self.osc_energy_binning = None
        self.binned_osc_probs = None
        self.binned_gradients = None
        self.binned_second_derivs = None
        if self.do_binned_osc_probs:
            self.osc_energy_binning = osc_energy_binning

            if self.oscillator is not None:
                self._prep_binned_osc(save_gradients=True)

        ## The weight that should be applied to events in this sample to
        ## recover the cross section that was used to generate the events
        self.fixed_xsec_weight = None

        ## The integrated flux for this subsample
        self.integrated_flux = None

        ## weight to apply to account for using less than max num of events in file
        self.n_max_events_weight = 1.0

    def _get_event_info(
        self,
        file: NuisanceFile,
        aux_vars: typing.List[str],
        progress_bar: bool,
        max_n_events: int = None,
    ) -> None:
        """read event info from input file and turn it into an array of events"""

        ## read arrays of particle info
        n_particle_array = file.get_array("nfsp")
        px = file.get_array("px")
        py = file.get_array("py")
        pz = file.get_array("pz")
        energies = file.get_array("E")
        pdg = file.get_array("pdg")

        ## read event level variables
        nu_pdg = file.get_array("PDGnu")
        nu_energies = file.get_array("Enu_true")
        modes = file.get_array("Mode")

        ## read auxilary variables specified by user
        aux_var_arrays = [file.get_array(aux_var) for aux_var in aux_vars]

        n_events_to_read = n_particle_array.shape[0]
        if max_n_events is not None:
            self.n_max_events_weight = n_events_to_read / min(
                max_n_events, n_events_to_read
            )

            n_events_to_read = min(max_n_events, n_events_to_read)

        iterable = range(n_events_to_read)
        if progress_bar:
            iterable = tqdm(
                range(n_events_to_read),
                desc=f"Reading events for subsample {self.name}",
            )

        for i_event in iterable:

            event = Event(
                e_nu=nu_energies[i_event], mode=modes[i_event], nu_pdg=nu_pdg[i_event]
            )

            event.add_particles_from_arrays(
                px=px[i_event],
                py=py[i_event],
                pz=pz[i_event],
                energies=energies[i_event],
                pdg=pdg[i_event],
            )

            ## add auxilary variables
            for aux_var, array in zip(aux_vars, aux_var_arrays):
                event.aux_vars[aux_var] = array[i_event]

            self.events.append(event)

    def fill_from_file(
        self,
        file: NuisanceFile,
        auxilary_variables=["q0", "q3", "Enu_QE"],
        progress_bar: bool = False,
        max_n_events: int = None,
    ) -> "SubSample":
        """Fill this subsample with events read in from a nuisance flat tree

        :param file: The path to the nuisance flat tree file
        :type file: NuisanceFile
        :param auxilary_variables: Values to store in the "aux_vars" (variables that are available to downstream analysis modules), defaults to ["q0", "q3", "Enu_QE"]
        :type auxilary_variables: list, optional
        :param progress_bar: If True, will display a progress bar showing how many events have been read, defaults to False
        :type progress_bar: bool, optional
        :param max_n_events: Max number of events to read from the file, if None then all events will be read, defaults to None
        :type max_n_events: int, optional
        :return: This SubSample object
        :rtype: SubSample
        """

        ## safely open the file
        with file as _file:
            self._get_event_info(
                _file,
                auxilary_variables,
                progress_bar=progress_bar,
                max_n_events=max_n_events,
            )

            self.flux_hist = _file.flux_hist.to_numpy()
            self.fixed_xsec_weight = _file.scale_factor

        self.flux_binning = Binning(["Enu_true"], bin_edges=[self.flux_hist[1]])

        self.integrated_flux = self.get_integrated_flux()

        return self

    def get_integrated_flux(
        self, bin_width_normalised: bool = False, scale_factor: float = 1 / 0.05
    ) -> float:
        """Get the integral of the flux histogram in this SubSample

        :param bin_width_normalised: if True, the bin contents will be multiplied by the bin width when taking the total. Equivalent to the "width" option in ROOT's TH1->Integral(), defaults to True
        :type bin_width_normalised: bool, optional
        :param scale_factor: Arbitrary scaling to apply to the flux. Default value of 1/50MeV is to account for T2K flux normalisation, defaults to 1/0.05
        :type scale_factor: float, optional
        :raises RuntimeError: If the flux histogram has not yet been initialised (i.e. the subsample has not been set up properly)
        :return: The integrated flux
        :rtype: float
        """

        if self.flux_hist is None:
            raise RuntimeError("hmmmm, flux hist is None. Has this subsample been initialised properly????")

        counts, bin_edges = (
            self.flux_hist
        )  ## counts are in units of [1 / (cm^2 * 50 MeV * 10^21 POT)]
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        ret = None

        if bin_width_normalised:
            ret = (
                counts * bin_widths
            ).sum()  ## flux in units of [1 / (cm^2 * 10^21 POT * 50MeV)]

        else:
            ret = counts.sum()

        return ret * scale_factor

    def get_xsec_weight(self) -> float:
        """Get the fixed cross section weight that should be applied to events - this is just the fScaleFactor from the nuisance files
        """

        return self.fixed_xsec_weight

    def get_pot_weight(self, pot: float) -> float:
        """Get the scaling that should be applied to events to approximate event rates for some target POT
        """

        return pot / self.base_pot

    def get_event_scaling(self) -> float:
        """Get the scaling that should be applied to events in this sub-sample to estimate event rates assuming the given target mass and POT
        """

        n_nucleons = self.parameters.target_material.get_n_nucleons(self.parameters.target_mass)
        pot_weight = self.get_pot_weight(self.parameters.pot)

        return (
            self.integrated_flux
            * self.fixed_xsec_weight
            * self.n_max_events_weight
            * n_nucleons
            * pot_weight
        )

    def _prep_binned_osc(self, save_gradients: bool = True, second_deriv: bool = False) -> None:
        """Prepare the arrays used for binned oscillation calculations
        """

        assert (
            self.oscillator is not None
        ), "trying to oscillate a subsample with no oscillator????!!!!????"

        energy_bin_centres = (
            self.osc_energy_binning[1:] + self.osc_energy_binning[:-1]
        ) / 2.0

        n_bins = energy_bin_centres.shape[0]

        self.binned_osc_probs = np.ones((n_bins,))

        self.binned_gradients = {}
        self.binned_second_derivs = {}

        for par_name in self.oscillator.parameters.keys():
            self.binned_gradients[par_name] = np.zeros((n_bins,))
            self.binned_second_derivs[par_name] = np.zeros((n_bins,))

        for i_bin in range(n_bins):

            osc_probs = self.oscillator.calculate_osc_probs(
                np.array([energy_bin_centres[i_bin]]), antineutrino=self.parameters.antinu
            )
            osc_prob_tensor = osc_probs.get_values(
                [0, self.parameters.initial_flavour, self.parameters.final_flavour]
            )

            self.binned_osc_probs[i_bin] = osc_prob_tensor.numpy()

            if save_gradients or second_deriv:

                for par_name, parameter in zip(
                    self.oscillator.parameters.keys(),
                    self.oscillator.parameters.values(),
                ):

                    grad_tensor = grad(osc_prob_tensor, parameter)
                    self.binned_gradients[par_name][i_bin] = grad_tensor.numpy()[0]

                    if second_deriv:

                        second_deriv_tensor = grad(grad_tensor, parameter)
                        self.binned_second_derivs[par_name][
                            i_bin
                        ] = second_deriv_tensor.numpy()

    def oscillate_events(
        self,
        progress_bar: bool = False,
        save_gradients: bool = False,
        second_deriv: bool = False,
    ) -> None:
        """Calculate oscillations for each event and fill auxilary variable "osc_weight" with tensor containing oscillation weight

        If there is no oscillator for this subsample then the oscillation weight will just be 1

        :param progress_bar: If True this will print a progress bar with info on how many events have been processed, defaults to False
        :type progress_bar: bool, optional
        :param save_gradients: If True, the gradient of the event weight wrt each oscillation parameter will be saved in the "osc_weight<PARAMETER NAME>_grad" aux variable, defaults to False
        :type save_gradients: bool, optional
        :param second_deriv: If True, the second derivative of the event weight wrt each oscillation parameter will be saved in the "osc_weight<PARAMETER NAME>_second_grad" aux variable, defaults to False
        :type second_deriv: bool, optional
        """

        ## TODO Move to OscillationCalculator along with binned oscillation stuff

        if self.oscillator is None:

            for event in self.events:
                event.aux_vars["osc_weight"] = 1.0

                for par_name in self.oscillator.parameters.keys():
                    if save_gradients:
                        event.aux_vars[f"osc_weight_{par_name}_grad"] = 0.0

                    if second_deriv:
                        event.aux_vars[f"osc_weight_{par_name}_second_grad"] = 0.0
            return

        iterator = self.events
        if progress_bar:
            iterator = tqdm(self.events, desc=f"oscillatin' events [{self.name}]")

        for event in iterator:

            if self.do_binned_osc_probs:

                event_e_bin = np.digitize(event.enu_true, self.osc_energy_binning)

                ## check for events outside of the osc energy range
                if (
                    event_e_bin < 0
                    or event_e_bin >= self.osc_energy_binning.shape[0] - 1
                ):

                    if self.parameters.initial_flavour == self.parameters.final_flavour:
                        event.aux_vars["osc_weight"] = 1.0
                    else:
                        event.aux_vars["osc_weight"] = 0.0

                    for par_name in self.oscillator.parameters.keys():
                        if save_gradients:
                            event.aux_vars[f"osc_weight_{par_name}_grad"] = 0.0
                        if second_deriv:
                            event.aux_vars[f"osc_weight_{par_name}_second_grad"] = 0.0

                    continue

                event.aux_vars["osc_weight"] = self.binned_osc_probs[event_e_bin]

                if save_gradients or second_deriv:

                    for par_name in self.oscillator.parameters.keys():

                        event.aux_vars[f"osc_weight_{par_name}_grad"] = (
                            self.binned_gradients[par_name][event_e_bin]
                        )

                        if second_deriv:

                            event.aux_vars[f"osc_weight_{par_name}_second_grad"] = (
                                self.binned_second_derivs[par_name][event_e_bin]
                            )

            else:

                osc_probs = self.oscillator.calculate_osc_probs(
                    np.array([event.enu_true]), antineutrino=self.antinu
                )
                event_weight = osc_probs.get_values(
                    [0, self.initial_flavour, self.final_flavour]
                )

                event.aux_vars["osc_weight"] = event_weight.numpy()

                if save_gradients or second_deriv:

                    for par_name, parameter in zip(
                        self.oscillator.parameters.keys(),
                        self.oscillator.parameters.values(),
                    ):

                        parameter_grad = grad(event_weight, parameter)

                        event.aux_vars[f"osc_weight_{par_name}_grad"] = (
                            parameter_grad.numpy()[0]
                        )

                        if second_deriv:

                            parameter_second_deriv = grad(parameter_grad, parameter)

                            event.aux_vars[f"osc_weight_{par_name}_second_grad"] = (
                                parameter_second_deriv.numpy()[0]
                            )

    def get_event_rate(
        self,
        binning: Binning,
        cut: typing.Callable = None,
        weight_var: str = None,
    ) -> np.ndarray:
        """Get binned event rate for this subsample in some particular binning

        :param binning: The binning to project into
        :type binning: Binning
        :param cut: A function describing a cut to apply to the events, defaults to None
        :type cut: typing.Callable, optional
        :param weight_var: The name of a variable to (stored in the "aux_vars") to apply as a weight when calculating the rates, defaults to None
        :type weight_var: str, optional
        :return: Array of event rates in the specified binning
        :rtype: np.ndarray
        """

        data_list = []

        ## we need to check for nan / None values as some variables might not be filled for some events
        energies = self.get_array("Enu_true", cut)
        not_nan = np.full(energies.shape, True)

        for iVar in range(binning.n_dims):
            array = self.get_array(binning.variables[iVar], cut)

            data_list.append(array)

            if array.shape[0] != 0:

                not_nan = np.logical_and(not_nan, np.logical_not(np.isnan(array)))

        ## caclulate oscillation weights if needed
        osc_weights = np.ones((np.sum(not_nan)))
        if self.oscillator is not None:
            osc_probs = self.oscillator.calculate_osc_probs(
                energies[not_nan], antineutrino=self.parameters.antinu
            )
            osc_weights = osc_probs.numpy()[:, self.parameters.initial_flavour, self.parameters.final_flavour]

        ## if weight variable specified make weight array
        weight_array = None
        if weight_var is not None:
            weight_array = self.get_array(weight_var, cut=cut)[not_nan]
        else:
            weight_array = np.ones((np.sum(not_nan)))

        ## now make the histogram
        hist, _ = np.histogramdd(
            [data[not_nan] for data in data_list],
            bins=binning.bin_edges,
            weights=osc_weights * weight_array,
        )

        return hist * self.get_event_scaling()


class Sample(SampleBase):

    def __init__(
        self,
        binning: Binning,
        subsamples: typing.List[SubSample],
        name: str,
    ):

        self.name: str = name
        self.binning: Binning = binning
        self.subsamples: typing.List[SubSample] = subsamples

        self.events: typing.List[Event] = []
        for subsample in self.subsamples:
            self.events += subsample.events

    def oscillate_events(
        self, progress_bar: bool = False, save_gradients: bool = False
    ) -> None:
        """Calculate oscillations for each subsample

        Just calls SubSample.oscillate_events() on each subsample
        """

        for subsample in self.subsamples:
            subsample.oscillate_events(
                progress_bar=progress_bar, save_gradients=save_gradients
            )

    def apply_selection(
        self, selection: 'SelectionBase', progress_bar: bool = False, strip_particle_info: bool = False
    ) -> "Sample":

        new_subsamples = []
        for subsample in self.subsamples:

            new_subsample = subsample.apply_selection(
                selection=selection, progress_bar=progress_bar, strip_particle_info=strip_particle_info
            )
            new_subsamples.append(new_subsample)

        new_sample = Sample(
            binning=self.binning,
            subsamples=new_subsamples,
            name=f"{self.name} [{selection.name}]",
        )

        return new_sample

    def imshow(
        self,
        axis,
        data_override: np.array,
        binning: Binning = None,
        z_label: str = None,
        **imshow_args,
    ):

        if binning is None:
            binning = self.binning

        assert binning.n_dims == 2, "need 2 dims for imshowing!!!"

        dat = data_override

        mappable = axis.pcolormesh(
            binning.bin_edges[0], binning.bin_edges[1], dat.T, **imshow_args
        )

        cbar = plt.colorbar(mappable)
        if z_label is None:
            cbar.set_label(
                f"N Events"
            )
        else:
            cbar.set_label(z_label)

        axis.set_title(f"{self.name}")

        plt.xlabel(binning.variables[0])
        plt.ylabel(binning.variables[1])

    def get_event_rates(
        self,
        binning: Binning = None,
        keep_zero=True,
        cut: typing.Callable = None,
        weight_var: str = None,
    ):

        if binning is None:
            binning = self.binning

        hist_total = np.zeros(binning.n_bins)

        for subsample in self.subsamples:

            hist_total += subsample.get_event_rate(
                binning,
                cut=cut,
                weight_var=weight_var,
            )

        if not keep_zero:
            hist_total[hist_total == 0] = np.nan

        return hist_total
