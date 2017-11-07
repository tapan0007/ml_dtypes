from .macrolayer import MacroLayer

class MacroPoolLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)


    #-----------------------------------------------------------------
    # void
    # compile_pool(FILE *out_binary,
    #        const addr_t ifmap_addr, const uint64_t ifmap_dims[4], // NCHW
    #        const uint64_t kernel_dims[4], // NCHW
    #        const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW
    #        const uint64_t stride_dims[4], // NCHW
    #        const ARBPRECTYPE dtype,
    #        POOLFUNC pool_func);
    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    def gIfmapString(self, layer):
        prevLayer   = layer.gPrevLayer(0)
        numBatches  = self.__NumBatches
        numIfmaps   = prevLayer.gNumOfmaps()
        ifmapWidth   = prevLayer.gOfmapWidth()
        ifmapHeight   = prevLayer.gOfmapHeight()

        ## const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
        ## N: batch size
        ## C: number of ifmaps / channels
        ## H: height of ifmap
        ## W: width of ifmap
        return (
            "ifmap_dims[0] = " + str(numBatches) + ";"
          + " ifmap_dims[1] = " + str(numIfmaps) + ";"
          + " ifmap_dims[2] = " + str(ifmapHeight) + ";"
          + " ifmap_dims[3] = " + str(ifmapWidth) + ";"
        )

    #-----------------------------------------------------------------
    def gKernelString(self, layer):
        numBatches  = self.__NumBatches
        kernelSize  = layer.gKernel()
        ifmapStride = self.__IfmapStride

        ## const uint64_t kernel_dims[4], // NCHW
        ## N: batch size
        ## C: number ifmaps / channels
        ## H: filter height
        ## W: filter width
        return (
            "kernel_dims[0] = " + str(numBatches)+ ";"
          + " kernel_dims[1] = " + str(ifmapStride) + ";"
          + " kernel_dims[2] = " + str(kernelSize) + ";"
          + " kernel_dims[3] = " + str(kernelSize) + ";"
        )

    #-----------------------------------------------------------------
    def gStrideString(self, layer):
        batchStride = self.__NumBatches = 1
        ifmapStride = 1
        ## const uint64_t stride_dims[4], // NCHW
        return (
            "pool_stride[0] = " + str(batchStride) + ";"
          + " pool_stride[1] = " + str(ifmapStride) + ";"
          + " pool_stride[2] = " + str(layer.gStride()) + ";"
          + " pool_stride[3] = " + str(layer.gStride()) + ";"
        )


    #-----------------------------------------------------------------
    def gOfmapAssertString(self, layer):
        assertStr   =  self.gMacroInstrGen().gAssertionStr()
        ofmapWidth   = layer.gOfmapWidth()
        ofmapHeight   = layer.gOfmapHeight()
        numOfmaps   = layer.gNumOfmaps()
        numBatches  = self.__NumBatches

        ##const addr_t ofmap_addr, uint64_t ofmap_dims[4],
        ## N: batch size
        ## C: number of ifmaps / channels
        ## H: height of ifmap
        ## W: width of ifmap
        return (
            assertStr + "(ofmap_dims[0] == " + str(numBatches) + ");"
          + " " + assertStr + "(ofmap_dims[1] == " + str(numOfmaps) + ");"
          + " " + assertStr + "(ofmap_dims[2] == " + str(ofmapHeight) + ");"
          + " " + assertStr + "(ofmap_dims[3] == " + str(ofmapWidth) + ");"
        )

    #-----------------------------------------------------------------
    def generatePool(self, poolType):
        layer = self.gLayer()
        f           = self.gFile()
        self.__NumBatches = 1    ## only one image in batch
        self.__BatchPoolDim = 1  ## no pooling across batches
        self.__BatchPoolStride = 1 ## same as above
        self.__IfmapDim = 1  ## no pooling across Ifmaps
        self.__IfmapStride = 1 ## same as above

        ind         = self.gIndent()

        ##
        s = [ "// " + layer.gName(),
              "pool_stride[1] = pool_stride[0] = " + str(layer.gStride()) + ";",
              self.gIfmapString(layer),
              self.gKernelString(layer),
              self.gStrideString(layer),

              "ifmap_addrs[0] = " + str(layer.gIfmapAddress()) + ";",
              "ofmap_addrs = " + str(layer.gOfmapAddress()) +";",
              "",
              "compile_pool(out_binary,",
              ind + "ifmap_addrs, ifmap_dims,",
              ind + "ofmap_address, ofmap_dims,",
              ind + "kernel_dims,",
              ind + "pool_stride,",
              ind + self.gMacroInstrGen().gDataTypeName() + ",",
              ind + "POOLFUNC::" + poolType + ");",
              "",

              self.gOfmapAssertString(layer),
            ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)

