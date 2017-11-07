from .macrolayer import MacroLayer

##########################################################
class MacroAddLayer(MacroLayer):
    """
    Generate macro instruction(s) for AddLayer
    """
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self):
        layer = self.gLayer()
        file = self.gFile()
        ind = self.gIndent()
        cmdlist = ["// " + layer.gName(),
                   "compile_add()",
                  ]
        cmds = ""
        for cmd in cmdlist:
            cmds += ind + cmd + "\n"

        file.write(cmds)

