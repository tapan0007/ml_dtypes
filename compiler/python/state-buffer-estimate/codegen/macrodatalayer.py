from .macrolayer import MacroLayer


##########################################################
class MacroDataLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        qq = '"'
        q  = "'"
        f = self.gFile()
        ind = self.gIndent()
        s = [ "// " + layer.gName(),
              "compile_read_ifmap(out_binary,",
              ind + str(layer.gOfmapAddress()) + ", ",
              ind + str(qq + layer.gInputDataFileName() + qq) +", "
              + qq + layer.gDataTensorDimSemantics() + qq + ");",
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)



