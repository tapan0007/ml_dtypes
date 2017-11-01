from .macrolayer import MacroLayer

class MacroPoolLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    def generatePool(self, layer, poolType):
        f           = self.gFile()
        ind         = self.gIndent()
        prevLayer   = layer.gPrevLayer(0)
        numIfmaps   = prevLayer.gNumOfmaps()
        ifmapSize   = prevLayer.gOfmapSize()
        kernelSize  = layer.gKernel()
        numOfmaps   = layer.gNumOfmaps()
        ofmapSize   = layer.gOfmapSize()
        numBatches  = 1
        batchStride = 1
        ifmapStride = 1
        assertStr   =  self.gMacroInstrGen().gAssertionStr()

        ##
        s = [ "// " + layer.gName(),
              "pool_stride[1] = pool_stride[0] = " + str(layer.gStride()) + ";",

              ## const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
              ## N: batch size
              ## C: number of ifmaps / channels
              ## H: height of ifmap
              ## W: width of ifmap
              (   "ifmap_dims[0] = " + str(numBatches) + ";"
                + " ifmap_dims[1] = " + str(numIfmaps) + ";"
                + " ifmap_dims[2] = " + str(ifmapSize) + ";"
                + " ifmap_dims[3] = " + str(ifmapSize) + ";"
              ),
              ## const uint64_t kernel_dims[4], // NCHW 
              ## N: batch size
              ## C: number ifmaps / channels
              ## H: filter height
              ## W: filter width
              (   "kernel_dims[0] = " + str(numBatches)+ ";"
                + " kernel_dims[1] = " + str(numIfmaps) + ";"
                + " kernel_dims[2] = " + str(kernelSize) + ";"
                + " kernel_dims[3] = " + str(kernelSize) + ";"
              ),

              ## const uint64_t stride_dims[4], // NCHW 
              (   "pool_stride[0] = " + str(batchStride) + ";"
                + " pool_stride[1] = " + str(ifmapStride) + ";"  
                + " pool_stride[2] = " + str(layer.gStride()) + ";"  
                + " pool_stride[3] = " + str(layer.gStride()) + ";"
              ),

              "",
              "compile_pool(out_binary,",
              ind + str(layer.gIfmapAddress()) + ", ifmap_dims,",
              ind + "kernel_dims,",
              ind + str(layer.gOfmapAddress()) + ", ofmap_dims,",
              ind + "pool_stride,",
              ind + self.gMacroInstrGen().gDataTypeName() + ",",
              ind + "POOLFUNC::" + poolType + ");",
              "",

              ##const addr_t ofmap_addr, uint64_t ofmap_dims[4], 
              ## N: batch size
              ## C: number of ifmaps / channels
              ## H: height of ifmap
              ## W: width of ifmap
              (
                  assertStr + "(ofmap_dims[0] == " + str(numBatches) + ");"
                + " " + assertStr + "(ofmap_dims[1] == " + str(numOfmaps) + ");"  
                + " " + assertStr + "(ofmap_dims[2] == " + str(ofmapSize) + ");"  
                + " " + assertStr + "(ofmap_dims[3] == " + str(ofmapSize) + ");"
              ),
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)

