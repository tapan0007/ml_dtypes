from .genpoollayer import GenPoolLayer


##########################################################
class GenAvgPoolLayer(GenPoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        self.generatePool(layer, "AVG_POOL")

