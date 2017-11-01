from .macrolayer import MacroLayer

##########################################################
class MacroAddLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        f = self.gFile()
        ind        = self.gIndent()
        s = [ "// " + layer.gName(),
              "compile_add()",
        ]
        ss = ""
        for x in s: ss += ind + x + "\n"

        f.write(ss)

