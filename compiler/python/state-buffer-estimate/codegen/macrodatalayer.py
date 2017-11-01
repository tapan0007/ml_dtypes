from .macrolayer import MacroLayer


##########################################################
class MacroDataLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        f = self.gFile()
        ind = self.gIndent()
        s = [ "// " + layer.gName(),
              "compile_read_ifmap(out_binary,",
              ind + str(layer.gOfmapAddress()) + ", " + str('"ifmap.py"') +",",
              ind + '"NCHW");',
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)



