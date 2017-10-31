from .genlayer import GenLayer

class GenPoolLayer(GenLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generatePool(self, layer, poolType):
        f = self.gFile()
        ind        = self.gIndent()
        prevLayer  = layer.gPrevLayer(0)
        numIfmaps  = prevLayer.gNumOfmaps()
        ifmapSize  = prevLayer.gOfmapSize()
        kernelSize = layer.gKernel()
        numOfmaps  = layer.gNumOfmaps()
        ofmapSize  = layer.gOfmapSize()

        s = [ "// " + layer.gName(),
              "stride[1] = stride[0] = " + str(layer.gStride()) + ";",
              (   "ifmaps_dims[0] = 1;"  ## num images
                + " ifmaps_dims[1] = " + str(ifmapSize) + ";"  ## image width?
                + " ifmaps_dims[2] = " + str(ifmapSize) + ";"  ## image height?
                + " ifmaps_dims[3] = " + str(numIfmaps) + ";"
              ),
              (   "kernel_dims[0] = " + str(numIfmaps)+ ";"
                + " kernel_dims[1] = " + str(kernelSize) + ";"  ## kernel width?
                + " kernel_dims[2] = " + str(kernelSize) + ";"  ## image height?
                + " kernel_dims[3] = " + str(numOfmaps) + ";"
              ),
              "",
              "compile_pool(out_binary,",
              ind + str(layer.gIfmapAddress()) + ", ifmap_dims,",
              ind + "kernel_dims,",
              ind + str(layer.gOfmapAddress()) + ", uint64_t ofmap_dims[4],",
              ind + "stride,   /* Height,Width */",
              ind + self.gMacroInstrGen().gDataTypeName() + ",",
              ind + poolType + ");",
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)

