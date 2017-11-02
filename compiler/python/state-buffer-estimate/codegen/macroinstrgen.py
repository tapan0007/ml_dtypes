
from nets.network import Network
from arch.arch import Arch

import layers.convlayer
import layers.maxpoollayer
import layers.avgpoollayer
import layers.addlayer

from .macroconvlayer    import MacroConvLayer
from .macromaxpoollayer import MacroMaxPoolLayer
from .macroavgpoollayer import MacroAvgPoolLayer
from .macrodatalayer    import MacroDataLayer
from .macroaddlayer     import MacroAddLayer

##########################################################
macro_instr_api=(
"""
/*****
void compile_read_ifmap(FILE *out_binary,
        const addr_t ifmap_sb_addr, const char *in_numpy_fname,
        const char *numpy_layout);

void compile_read_filter(FILE *out_binary,
        const addr_t filter_sb_addr, const char *in_numpy_fname, 
        const char *numpy_layout);

void compile_write_ofmap(FILE *out_binary,
        const char *out_numpy_name, const addr_t ofmap_sb_addr,
        const uint64_t dims[4],
        const size_t word_size);

// *[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays 
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the 
// * array is the wrap offset 

void
compile_convolve(FILE *out_binary,
        const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], // NCHW
        const addr_t *filter_addr, const uint64_t filter_dims[4], // MCRS 
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW 
        const ARBPRECTYPE dtype,
        const uint8_t padding[2],  // Height,Width 
        const uint8_t stride[2],   // Height,Width 
        const uint8_t dilate[2]);  // Height,Width 

void
compile_pool(FILE *out_binary,
        const addr_t ifmap_addr, const uint64_t ifmap_dims[4], // NCHW 
        const uint64_t kernel_dims[4], // NCHW 
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output NCHW 
        const uint64_t stride_dims[4], // NCHW 
        const ARBPRECTYPE dtype,
        POOLFUNC pool_func);
*****/


"""
)
 

##########################################################
class MacroInstrGen(object):
    #-----------------------------------------------------------------
    def createGenMap(self):
        self.__Map = {
            layers.datalayer.DataLayer       : MacroDataLayer(self),
            layers.convlayer.ConvLayer       : MacroConvLayer(self),
            layers.maxpoollayer.MaxPoolLayer : MacroMaxPoolLayer(self),
            layers.avgpoollayer.AvgPoolLayer : MacroAvgPoolLayer(self),
            #layers.addlayer.AddLayer         : MacroAddLayer(self),
        }

    #-----------------------------------------------------------------
    def __init__(self, ntwk, arch):
        self.__Network = ntwk
        self.__Arch = arch
        self.__Indent = "    "
        self.__AssertionStr = "Assert"

        self.createGenMap()

    def gDataTypeName(self):
        return "ARBPRECTYPE::" + self.__Network.gDataType().gName().upper()

    def gIndent(self):
        return self.__Indent

    def gAssertionStr(self):
        return self.__AssertionStr

    #-----------------------------------------------------------------
    def gGenFunc(self, layer):
        try:
            generator = self.__Map[layer.__class__]
        except KeyError:
            generator = None
        return generator

    #-----------------------------------------------------------------
    def generate(self, fileName):
        with open(fileName, "w") as f:
            self.__File = f
            self.generateFile()

    def gFile(self):
        return self.__File

    #-----------------------------------------------------------------
    def writeIfc(self):
        f = self.__File
        #f.write(macro_instr_api)

    def writeIncludes(self):
        f = self.__File
        f.write('\n')
        f.write('#include "cnpy.h"\n')
        f.write('#include "tpb_isa.h"\n')
        f.write('#include "uarch_cfg.h"\n')
        f.write('#include "tcc.h"\n')

    def writeDefines(self):
        f = self.__File
        f.write('#define Assert(X) assert(X)\n')

    #-----------------------------------------------------------------
    def generateFile(self):
        nl = "\n"
        self.writeIfc()
        self.writeIncludes()
        self.writeDefines()
        f = self.__File


        ####################################################
        ind = self.gIndent()
        ind2 = ind*2
        sep = "//-----------------------------------------" + nl

        ####################################################
        header = [
            nl,
            "int",
            "main(int argc, char* argv[])",
            "{",
            ind + "if (argc < 2) {",
            ind2 +'    fprintf(stderr, "Usage: %s out_obj_file\\n", argv[0]);',
            ind2 +"    exit(1);",
            ind + "}",
            ind + 'FILE* const out_binary = fopen(argv[1], "w");',
            ind + "if (! out_binary) {",
            ind2 +'    fprintf(stderr, "Cannot open file %s\\n", argv[1]);',
            ind2 +"    exit(1);",
            ind + "}",
            "",
            ind + "uint64_t ofmap_dims[4];",
            ind + "addr_t   ifmap_addrs[1];",   ## temporary for single Ifmap
            ind + "uint64_t ifmap_dims[4];",
            ind + "addr_t   filter_addr[1];",   ## temporary for single Ifmap
            ind + "uint64_t filter_dims[4];",
            ind + "uint64_t kernel_dims[4];",
            ind + "uint64_t pool_stride[4];",
            ind + "uint8_t  convolve_stride[2];",
            ind + "uint8_t padding[2];",
            ind + "uint8_t dilate[2];",
            "",
        ]
        for l in header:
            f.write(l+nl)

        ####################################################
        for layer in self.__Network.gSchedLayers():
            generator = self.gGenFunc(layer)
            if generator:
                f.write(ind + sep)
                generator.generate(layer)

        ####################################################
        footer = [
            "",
            ind + sep,
            ind + "fclose(out_binary);",
            ind + "return 0;",
            "}",
            "",
        ]
        for l in footer:
            f.write(l+nl)

