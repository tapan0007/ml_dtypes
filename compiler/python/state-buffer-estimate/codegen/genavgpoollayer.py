from .genlayer import GenLayer


##########################################################
class GenAvgPoolLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        print("compile_avgpool", layer)


