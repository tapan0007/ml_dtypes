from .genlayer import GenLayer


##########################################################
class GenConvLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generate(self, layer):
        f = self.gFile()
        ind        = self.gIndent()
        prevLayer  = layer.gPrevLayer(0)
        numIfmaps  = prevLayer.gNumOfmaps()
        ifmapSize  = prevLayer.gOfmapSize()
        numOfmaps  = layer.gNumOfmaps()
        ofmapSize  = layer.gOfmapSize()
        kernelSize = layer.gKernel()

        s = [ "// " + layer.gName(),
              "stride[1] = stride[0] = " + str(layer.gStride()) + ";",
              "padding[1] = padding[0] = 0;",
              "dilate[1] = dilate[0] = 0;",
              (   "ifmaps_dims[0] = 1;"  ## num images
                + " ifmaps_dims[1] = " + str(ifmapSize) + ";"  ## image width?
                + " ifmaps_dims[2] = " + str(ifmapSize) + ";"  ## image height?
                + " ifmaps_dims[3] = " + str(numIfmaps) + ";"
              ),
              (   "filter_dims[0] = " + str(numIfmaps)   + ";"  ## num images
                + " filter_dims[1] = " + str(kernelSize) + ";"  ## image width?
                + " filter_dims[2] = " + str(kernelSize) + ";"  ## image height?
                + " filter_dims[3] = " + str(numOfmaps)  + ";"
              ),
              "",
              "compile_convolve(out_binary,",
              ind + str(layer.gIfmapAddress())  + ", ifmap_dims,",
              ind + str(layer.gWeightAddress()) + ", filter_dims,",
              ind + str(layer.gOfmapAddress())  + ", ofmap_dims,",
              ind + self.gMacroInstrGen().gDataTypeName() + ",",
              ind + "padding,",
              ind + "stride,",
              ind + "dilate);",
           ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)

