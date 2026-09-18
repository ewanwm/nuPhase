"""Base module classes

User modules should inherit from these classes in order to be usable in 
other parts of the code
"""

import abc
import typing

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nuPhase.sample import Sample
    from nuPhase.event import Event

class ModuleBase(abc.ABC):
    """Base class of all modules
    """

class TransformationBase(ModuleBase):
    """All transformation modules should inherit from this

    Your transformation *must* implement the _apply(self, event) function.
    This is where the actual event level transformation stuff goes.

    If your transformation requires any sample level information, this should 
    be set up in the optional _initialise(self, sample) method. You should then implement
    the functionality to reset this in the _finalise(self, sample) method.
    """

    def _initialise(self, sample: 'Sample') -> None:
        """Set up for the transformation should go here

        :param sample: The sample that the transformation is being run on
        :type sample: 'Sample'
        """

        pass

    def initialise(self, sample: 'Sample') -> None:
        """Initialise the transformation
        """

        self.sample = sample

        ## call the user implemented initialisation function 
        self._initialise(sample)
        
    @abc.abstractmethod
    def _apply(self, event: 'Event') -> None:
        """The actual event level transformation should go here 

        :param event: _description_
        :type event: 'Event'
        """

        raise NotImplementedError()

    def apply(self, event: 'Event') -> None:
        """Apply the transformation to an event

        :param event: The event to apply the transformation to
        :type event: 'Event'
        """

        ## call the user implemented apply function
        self._apply(event)

    def _finalise(self, sample: 'Sample') -> None:
        """Any teardown of the transformation should go here

        :param sample: The sample this transformation is being applied to 
        :type sample: 'Sample'
        """

        pass

    def finalise(self, sample: 'Sample') -> None:
        """Tear down the transformation

        :param sample: The sample that the transformation was being applied to
        :type sample: 'Sample'
        """

        ## call user implemented finalise function
        self._finalise(sample)

        ## unset the sample 
        self.sample = None


class SelectionBase(TransformationBase):

    @abc.abstractmethod
    def _apply(self, event: 'Event') -> bool:
        """Implement your selection code here

        :param event: The event that the selection is being applied to
        :type event: 'Event'
        :return: should return true if the event passed the selection or false otherwise
        :rtype: bool
        """

        raise NotImplementedError()

    def apply(self, event: 'Event') -> bool:
        """Apply the selection to an event

        :param event: The event to apply the selection to
        :type event: 'Event'
        :return: True if the event passed the selection or false otherwise
        :raises TypeError: If the user selection code does not return a boolean
        :rtype: bool
        """

        ## call the user implemented apply function
        ret = self._apply(event)

        if type(ret) is not bool:
            raise TypeError("User selection code apply() method returned a non-bool value!!")

        return ret


class AnalysisBase(ModuleBase):
    """Any analysis module should inherit from this base class
    """

    @abc.abstractmethod
    def run() -> None:

        raise NotImplementedError()

    def initialise() -> None:
        """Any initialisation of the analysis should go here. 
        e.g. opening output files, reading inputs etc.
        """

        pass

    def finalise() -> None:
        """Any teardown of the analysis should go here.

        e.g. closing output or input files 
        """

        pass
