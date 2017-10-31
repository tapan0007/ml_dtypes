from abc             import ABCMeta, abstractmethod

##########################################################
class GenLayer(object, metaclass = ABCMeta):
    def __init__(self, macroInstrGen):
        self.__MacroInstrGen = macroInstrGen
        self.__Indent = "    "

    def gMacroInstrGen(self):
        return self.__MacroInstrGen

    def gFile(self):
        return self.__MacroInstrGen.gFile()

    def gIndent(self):
        return self.__MacroInstrGen.gIndent()

    @abstractmethod
    def generate(self, layer):
        assert(False)

