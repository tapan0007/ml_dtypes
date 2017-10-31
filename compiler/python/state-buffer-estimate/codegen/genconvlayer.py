from .genlayer import GenLayer


##########################################################
class GenConvLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        print("compile_convolve", layer)

