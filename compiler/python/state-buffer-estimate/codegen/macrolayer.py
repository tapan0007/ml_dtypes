from abc             import ABCMeta, abstractmethod

##########################################################
class MacroLayer(object, metaclass = ABCMeta):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        self.__MacroInstrGen = macroInstrGen
        self.__Indent = "    "

    #-----------------------------------------------------------------
    def gMacroInstrGen(self):
        return self.__MacroInstrGen

    #-----------------------------------------------------------------
    def gFile(self):
        return self.__MacroInstrGen.gFile()

    #-----------------------------------------------------------------
    def gIndent(self):
        return self.__MacroInstrGen.gIndent()

    #-----------------------------------------------------------------
    @abstractmethod
    def generate(self):
        assert(False)

    #-----------------------------------------------------------------
    def gLayer(self):
        return self.__Layer

    #-----------------------------------------------------------------
    def rLayer(self, layer):
        self.__Layer = layer

    #-----------------------------------------------------------------
    def gWriteOfmapStatement(self, ind):
        layer = self.gLayer()
        nl = "\n"
        qq = '"'
        nn = layer.gNetwork()
        layerFileName = nn.gName().lower() + "-" + layer.gName().lower() + "-simout.npy"
        layerFileName = layerFileName.replace("/", "-")

        lines = [
            "ofmap_addrs = " + str(layer.gOfmapAddress()) + ";",
            "compile_write_ofmap(out_binary, ",
            ind + qq + layerFileName + qq + ", ",
            ind + "ofmap_addrs , ofmap_dims, " + "ARBPRECTYPE::" + layer.gDataType().gTccName() + ");",
        ]
        return lines


