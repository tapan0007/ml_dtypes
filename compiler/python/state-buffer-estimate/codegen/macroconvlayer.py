from .macrolayer import MacroLayer


##########################################################
class MacroConvLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    # *[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
    # * deal with cases iwhen with the number of ifmap channels is > number of rows.
    # * In this case, the ifmaps and filters must be "wrapped".  Each address in the
    # * array is the wrap offset
    #
    # void
    # compile_convolve(FILE *out_binary,
    #         const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
    #         const addr_t *filter_addr, const uint64_t filter_dims[4], // MCRS
    #         const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW
    #         const ARBPRECTYPE in_dtype, const ARBPRECTYPE out_dtype,
    #         const uint8_t padding[2],  // Height,Width
    #         const uint8_t stride[2],   // Height,Width
    #         const uint8_t dilate[2]);  // Height,Width
    #-----------------------------------------------------------------
    def generate(self):
        layer = self.gLayer()
        assert layer
        qq = '"'
        q = "'"
        f = self.gFile()
        ind        = self.gIndent()
        prevLayer  = layer.gPrevLayer(0)
        numIfmaps  = prevLayer.gNumOfmaps()
        ifmapWidth  = prevLayer.gOfmapWidth()
        ifmapHeight  = prevLayer.gOfmapHeight()
        numOfmaps  = layer.gNumOfmaps()
        ofmapWidth  = layer.gOfmapWidth()
        ofmapHeight  = layer.gOfmapHeight()
        kernelHeight = layer.gKernelHeight()
        kernelWidth = layer.gKernelWidth()
        numBatches = 1
        assertStr  =  self.gMacroInstrGen().gAssertionStr()

        ##
        ##
        s = ["// convolution: " + layer.gName(),
             ## const addr_t *filter_addr, const uint64_t filter_dims[4], // MCRS
             ## M: number of ofmaps
             ## C: number ifmaps / channels
             ## R: filter height
             ## S: filter width
             "filter_addr[0] = " + str(layer.gWeightAddress()) + ";",
             "filter_file_names[0] = " + qq + layer.gFilterFileName() + qq + ";",
             "",

             "compile_read_filter(out_binary, filter_addr[0], filter_file_names[0], "
                   + qq + layer.gFilterTensorDimSemantics() + qq +  ");",

             "",

             ## const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
             ## N: batch size
             ## C: number of ifmaps / channels
             ## H: height of ifmap
             ## W: width of ifmap
             "ifmap_addrs[0]     = " + str(layer.gIfmapAddress()) + ";",

             "ifmap_dims[IfmapIndex_N]      = " + str(numBatches) + ";",
             "ifmap_dims[IfmapIndex_C]      = " + str(numIfmaps) + ";",
             "ifmap_dims[IfmapIndex_H]      = " + str(ifmapHeight) + ";",
             "ifmap_dims[IfmapIndex_W]      = " + str(ifmapWidth) + ";",

             "// filter_addr",
             "filter_dims[FilterIndex_M]     = " + str(numOfmaps)   + ";",
             "filter_dims[FilterIndex_C]     = " + str(numIfmaps) + ";",  
             "filter_dims[FilterIndex_R]     = " + str(kernelHeight) + ";",
             "filter_dims[FilterIndex_S]     = " + str(kernelWidth)  + ";",

             "ofmap_addrs        = " + str(layer.gOfmapAddress()) + ";",
             "// ofmap_dims (output)",
             "// precision",
             "padding[0]         = " + str(layer.gPaddingRight()) + ";",
             "padding[1]         = " + str(layer.gPaddingTop()) + ";",
             "convolve_stride[0] = " + str(layer.gStrideLR()) + ";",
             "convolve_stride[1] = " + str(layer.gStrideBT()) + ";",
             "dilate[0]          = 0;",
             "dilate[1]          = 0;",
             "",

             "compile_convolve(out_binary,",
             ind + "ifmap_addrs, ifmap_dims,",
             ind + "filter_addr, filter_dims,",
             ind + "ofmap_addrs, ofmap_dims,",
             ind + self.gMacroInstrGen().gDataTypeName(layer) + ",",
             ind + self.gMacroInstrGen().gDataTypeName(layer) + ",",
             ind + "padding,",
             ind + "convolve_stride,",
             ind + "dilate);",

             "",
             ## const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW
             ## N: batch size
             ## C: number of ifmaps / channels
             ## H: height of ofmap
             ## W: width of ofmap
             assertStr + "(ofmap_dims[OfmapIndex_N] == " + str(numBatches) + ");",
             assertStr + "(ofmap_dims[OfmapIndex_C] == " + str(numOfmaps) + ");",
             assertStr + "(ofmap_dims[OfmapIndex_H] == " + str(ofmapHeight) + ");",
             assertStr + "(ofmap_dims[OfmapIndex_W] == " + str(ofmapWidth) + ");",
             "",
            ]

        for x in self.gWriteOfmapStatement(ind):
            s.append(x)

        ss = ""
        for x in s:
            if x != "":
                ss += ind + x + "\n"
            else:
                ss += x + "\n"
        f.write(ss)

