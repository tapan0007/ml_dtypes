from .macropoollayer import MacroPoolLayer


##########################################################
class MacroAvgPoolLayer(MacroPoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        self.generatePool(layer, "AVG_POOL")

