from abc             import ABCMeta, abstractmethod

##########################################################
class GenLayer(object, metaclass = ABCMeta):
    def __init__(self, macroInstrGen):
        self.__MacroInstrGen = macroInstrGen

    @abstractmethod
    def generate(self, layer):
        assert(False)

