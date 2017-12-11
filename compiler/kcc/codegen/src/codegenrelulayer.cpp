from .macroactivlayer import MacroActivLayer


##########################################################
class MacroReluLayer(MacroActivLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    @classmethod
    def gActivFunc(self):
        return "RELU"


