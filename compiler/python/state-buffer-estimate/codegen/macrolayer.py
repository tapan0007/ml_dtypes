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
    def generate(self, layer):
        assert(False)

    #-----------------------------------------------------------------
    def gWriteOfmapStr(self, file, ind):
        nl = "\n"
        qq = '"'

        ind + 
        lines = (
            ofmap_addrsStr = "ofmap_addrs = " + str(self.gOfmapsAddress());
            "compile_write_ofmap(out_binary, "
            + qq + self.__Network.gName().lower() + "-" + self.gName().lower + "-out.npy" + qq + ", "
            + ofmap_addrsStr + ", "
            + "ARBPRECTYPE::" + self.__Network.gDataType().gTccName()
            + ");",

        f.write(l+nl)

