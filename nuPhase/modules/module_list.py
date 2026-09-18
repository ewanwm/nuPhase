import typing
from enum import IntEnum
import inspect
import sys

from nuPhase.modules.base import ModuleBase, TransformationBase, AnalysisBase

## import the modules containing nuPhase module objects
from nuPhase.modules import analysis 
from nuPhase.modules.analysis import *
from nuPhase.modules import transformations
from nuPhase.modules.transformations import *
from nuPhase.modules import selection
from nuPhase.modules.selection import *

class moduleTypeEnum(IntEnum):
    """Types of modules
    """

    analysis = 0
    transformation = 1
    selection = 2

class Singleton(type):
    """Singleton base class
    
    Using this as a base class ensures that there will only ever be one of the derived object.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class ModuleList(metaclass=Singleton):

    def __init__(self):
        
        self._modules: typing.List[typing.Type[ModuleBase]] = []
        self._module_map: typing.List[str, typing.Type[ModuleBase]] = {}

        self._analysis_modules: typing.List[typing.Type[AnalysisBase]] = []
        self._transformation_modules: typing.List[typing.Type[TransformationBase]] = []
        self._selection_modules: typing.List[typing.Type[SelectionBase]] = []
    
    def register(self, module: typing.Type[ModuleBase]) -> None:
        """Register a module

        :param module: The module class to register
        :type module: typing.Type[ModuleBase]
        """

        if not issubclass(module, ModuleBase):
            raise TypeError(f"module {module.__name__} is not derived from ModuleBase!! cannot be registered as a module")
        
        if module in self._modules:
            print(f"WARNING: Trying to register module {module.__name__} but it has already been registered")
            return
        
        self._modules.append(module)
        self._module_map[module.__name__] = self._modules[-1]

        ## set the module_type 
        if issubclass(module, TransformationBase):

            ## all selections are transformations so have to check what kind of transformation this is
            if issubclass(module, SelectionBase):
                self._selection_modules.append(module)
            else:
                self._transformation_modules.append(module)

        elif issubclass(module, AnalysisBase):
            self._analysis_modules.append(module)

        else:
            raise TypeError("??")

    def get_module_type(self, module: typing.Type[ModuleBase]) -> moduleTypeEnum:
        """Get the type of a particular module

        :param module: The module
        :type module: typing.Type[ModuleBase]
        :raises ValueError: if the module has not been registered with the ModuleList manager
        :return: the type of the module
        :rtype: moduleTypeEnum
        """

        if module not in self._modules:
            raise ValueError(f"Unknown module: {module.__name__}")

        if module in self._analysis_modules:
            return moduleTypeEnum.analysis
        elif module in self._transformation_modules:
            return moduleTypeEnum.transformation
        elif module in self._selection_modules:
            return moduleTypeEnum.selection
        else:
            raise ValueError("Unknown module type???")

    def get_modules(self) -> typing.List[typing.Type[ModuleBase]]:
        """Get a list of all the registered modules

        :return: All the registered modules
        :rtype: typing.List[typing.Type[ModuleBase]]
        """

        return self._modules
    
    def get_module_names(self) -> typing.List[str]:
        """Get a list of the names of all registered modules

        :return: list of names
        :rtype: typing.List[str]
        """

        return [c.__name__ for c in self._modules]
    
    def get_module(self, name: str) -> typing.Type[ModuleBase]:
        """Get a module by name

        :param name: the name of the module
        :type name: str
        :raises ValueError: If the module that has been asked for has not been registered
        :return: The requested module
        :rtype: typing.Type[ModuleBase]
        """

        if not name in self._module_map.keys():
            raise ValueError(f"Asked for module {name} but it has not been registered!!\nAvailable modules: {self._modules}")
        
        return self._module_map[name]

    
####### Register all the modules ###########

## get all module classes
classes = []
for module in ['nuPhase.modules.analysis', 'nuPhase.modules.transformations', 'nuPhase.modules.selection']:
    ## this will add all ModuleBase derived classes from the specified module
    classes += [cls_obj for _, cls_obj in inspect.getmembers(sys.modules[module]) if inspect.isclass(cls_obj) and issubclass(cls_obj, ModuleBase)]

for class_obj in classes:
    ModuleList().register(class_obj)
    