from .macropoollayer import MacroPoolLayer

##########################################################
class MacroMaxPoolLayer(MacroPoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        self.generatePool(layer, "MAX_POOL")

