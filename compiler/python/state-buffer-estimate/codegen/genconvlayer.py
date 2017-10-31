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
        numBatches = 1
        assertStr   =  self.gMacroInstrGen().gAssertionStr()

        ##
        ##
        s = [ "// " + layer.gName(),
              "convolve_stride[1] = convolve_stride[0] = " + str(layer.gStride()) + ";",
              "padding[1] = padding[0] = 0;",
              "dilate[1] = dilate[0] = 0;",

              ## const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
              ## N: batch size
              ## C: number of ifmaps / channels
              ## H: height of ifmap
              ## W: width of ifmap
              (   "ifmaps_dims[0] = " + str(numBatches) + ";"  ## num images
                + " ifmaps_dims[1] = " + str(numIfmaps) + ";"  ## image width?
                + " ifmaps_dims[2] = " + str(ifmapSize) + ";"  ## image height?
                + " ifmaps_dims[3] = " + str(ifmapSize) + ";"
              ),
              ## const addr_t *filter_addr, const uint64_t filter_dims[4], // MCRS
              ## M: number of ofmaps
              ## C: number ifmaps / channels
              ## R: filter height
              ## S: filter width
              (   "filter_dims[0] = " + str(numOfmaps)   + ";"  ## num images
                + " filter_dims[1] = " + str(numIfmaps) + ";"  ## image width?
                + " filter_dims[2] = " + str(kernelSize) + ";"  ## image height?
                + " filter_dims[3] = " + str(kernelSize)  + ";"
              ),
              "",
              "compile_convolve(out_binary,",
              ind + str(layer.gIfmapAddress())  + ", ifmap_dims,",
              ind + str(layer.gWeightAddress()) + ", filter_dims,",
              ind + str(layer.gOfmapAddress())  + ", ofmap_dims,",
              ind + self.gMacroInstrGen().gDataTypeName() + ",",
              ind + "padding,",
              ind + "convolve_stride,",
              ind + "dilate);",
              "",
              ## const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW 
              ## N: batch size
              ## C: number of ofmaps / channels
              ## H: height of ofmap
              ## W: width of ofmap
              (
                  assertStr + "(ofmaps_dims[0] == " + str(numBatches) + ");"
                + " " + assertStr + "(ofmaps_dims[1] == " + str(numOfmaps) + ");"  
                + " " + assertStr + "(ofmaps_dims[2] == " + str(ofmapSize) + ");"  
                + " " + assertStr + "(ofmaps_dims[3] == " + str(ofmapSize) + ");"
              ),
           ]

        ss = ""
        for x in s: ss += ind + x + "\n"
        f.write(ss)

