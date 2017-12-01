from .macrolayer import MacroLayer


##########################################################
class MacroReluLayer(MacroLayer):
    #-----------------------------------------------------------------
    def __init__(self, macroInstrGen):
        super().__init__(macroInstrGen)

    #-----------------------------------------------------------------
    # *[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
    # * deal with cases iwhen with the number of ifmap channels is > number of rows.
    # * In this case, the ifmaps and filters must be "wrapped".  Each address in the
    # * array is the wrap offset
    #
    #  void
    #  compile_activation(FILE *out_binary,
    #            const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
    #            const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
    #            const ARBPRECTYPE in_dtype,
    #            const ARBPRECTYPE out_dtype,
    #            ACTIVATIONFUNC act_func);
    #-----------------------------------------------------------------
    def generate(self):
        print("Generating RELU compile instr")
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
        numBatches = 1
        assertStr  =  self.gMacroInstrGen().gAssertionStr()

        inDataType  = self.gMacroInstrGen().gDataTypeName(layer)
        outDataType = self.gMacroInstrGen().gDataTypeName(layer)
        ##
        ##
        s = ["// relu: " + layer.gName(),
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

             "ofmap_addrs        = " + str(layer.gOfmapAddress()) + ";",
             "// ofmap_dims (output)",
             "// precision",

             "compile_activation(out_binary,",
             ind + "ifmap_addrs[0], ifmap_dims,",
             ind + "ofmap_addrs, ofmap_dims,",
             ind + inDataType + ",",
             ind + outDataType + ",",
             ind + "ACTIVATIONFUNC::RELU);",


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

