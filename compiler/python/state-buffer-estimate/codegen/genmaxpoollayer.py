from .genpoollayer import GenPoolLayer

##########################################################
class GenMaxPoolLayer(GenPoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        self.generatePool(layer, "MAX_POOL")

