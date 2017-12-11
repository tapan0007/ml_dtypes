from .macroactivlayer import MacroActivLayer


##########################################################
class MacroTanhLayer(MacroActivLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    @classmethod
    def gActivFunc(self):
        return "TANH"



