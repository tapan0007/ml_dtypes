from .genlayer import GenLayer

##########################################################
class GenMaxPoolLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        print("compile_maxpool", layer)

