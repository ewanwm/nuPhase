"""Base module classes

User modules should inherit from these classes in order to be usable in 
other parts of the code
"""

import abc
import typing
from argparse import ArgumentParser, Namespace

from nuPhase.sample import Sample
from nuPhase.event import Event

class ModuleBase(abc.ABC):
    """Base class of all modules
    """

    def help(self) -> str:
        """Override this to print a useful help message in the CLI

        :return: handy help message
        :rtype: str
        """

        return ""

    def setup_parser(self, parser: ArgumentParser) -> None:
        """Sets up command line interface parser options

        :param parser: The parser that options will be added to
        :type parser: ArgumentParser
        """

        ## call user defined parser code
        self._setup_parser(parser)

    @abc.abstractmethod
    def _setup_parser(self, parser: ArgumentParser) -> None:
        """Put your code to add command line interface parser options here

        :param parser: The parser that options will get added to
        :type parser: ArgumentParser
        """

        raise NotImplementedError()

    def parse_args(self, args: Namespace) -> None:
        """Takes arguments that are defined by setup_parser and convert them into class members

        :param args: The parsed command line arguments
        :type args: Namespace
        """

        self._parse_args(args)

    @abc.abstractmethod
    def _parse_args(self, args: Namespace) -> None:
        """Code to take parsed command line arguments and turn them into useful internal class variables should go here

        :param args: parsed command line arguments
        :type args: Namespace
        """

        raise NotImplementedError

class TransformationBase(ModuleBase):
    """All transformation modules should inherit from this

    Your transformation *must* implement the _apply(self, event) function.
    This is where the actual event level transformation stuff goes.

    If your transformation requires any sample level information, this should 
    be set up in the optional _initialise(self, sample) method. You should then implement
    the functionality to reset this in the _finalise(self, sample) method.
    """

    def setup_parser(self, parser: ArgumentParser) -> None:
        """Set up command line parser options

        :param parser: CLI parser
        :type parser: ArgumentParser
        """

        ## set up any arguments from base class
        super().setup_parser(parser)
    
        parser.add_argument('--input-sample', '-i', help="Path to the sample that the transformation should be applied to", required=True, type=str)
        parser.add_argument('--progress', '-p', help="Show progress bar", action="store_true", required=False)

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
    def run(self) -> None:

        raise NotImplementedError()

    def initialise(self) -> None:
        """Set up the module
        """

        ## call user specified code
        self._initialise()

    def finalise(self) -> None:
        """Tear down the module
        """

        ## call user specified code
        self._finalise()

    def _initialise(self) -> None:
        """Any initialisation of the analysis should go here. 
        e.g. opening output files, reading inputs etc.
        """

        pass

    def _finalise(self) -> None:
        """Any teardown of the analysis should go here.

        e.g. closing output or input files 
        """

        pass

    def setup_parser(self, parser):

        super().setup_parser(parser)

        parser.add_argument('--fd-samples', nargs='+', default=[], help="list of far detector samples to consider", required=True)
        parser.add_argument('--nd-samples', nargs='+', default=[], help="list of near detector samples to consider", required=True)

    def parse_args(self, args):

        super().parse_args(args)
    
        self.nd_samples = [Sample.from_file(file_name) for file_name in args.nd_samples]
        self.fd_samples = [Sample.from_file(file_name) for file_name in args.fd_samples]

