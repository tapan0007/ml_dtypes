
from nets.network import Network
from arch.arch import Arch

import layers.convlayer
import layers.maxpoollayer
import layers.avgpoollayer
import layers.addlayer

from .genconvlayer    import GenConvLayer
from .genmaxpoollayer import GenMaxPoolLayer
from .genavgpoollayer import GenAvgPoolLayer
from .gendatalayer    import GenDataLayer
from .genaddlayer     import GenAddLayer

##########################################################
class MacroInstrGen(object):
    #-----------------------------------------------------------------
    def createGenMap(self):
        self.__Map = {
            layers.datalayer.DataLayer       : GenDataLayer(self),
            layers.convlayer.ConvLayer       : GenConvLayer(self),
            layers.maxpoollayer.MaxPoolLayer : GenMaxPoolLayer(self),
            layers.avgpoollayer.AvgPoolLayer : GenAvgPoolLayer(self),
            layers.addlayer.AddLayer        : GenAddLayer(self),
        }

    #-----------------------------------------------------------------
    def __init__(self, ntwk, arch):
        self.__Network = ntwk
        self.__Arch = arch
        self.__Indent = "    "

        self.createGenMap()

    def gDataTypeName(self):
        return self.__Network.gDataType().gName().upper()

    def gIndent(self):
        return self.__Indent

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
        info = (
        """
/*****************************************************************************
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

// [ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
// deal with cases iwhen with the number of ifmap channels is > number of rows.
// In this case, the ifmaps and filters must be "wrapped".  Each address in the
// array is the wrap offset

void
compile_convolve(FILE *out_binary,
    const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4],
    const addr_t *filter_addr, const uint64_t filter_dims[4],
    const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output
    const ARBPRECTYPE dtype,
    const uint8_t padding[2],  // Height,Width
    const uint8_t stride[2],   // Height,Width
    const uint8_t dilate[2]);  // Height,Width

void
compile_pool(FILE *out_binary,
    const addr_t ifmap_addr, const uint64_t ifmap_dims[4],
    const uint64_t kernel_dims[4],
    const addr_t ofmap_addr, uint64_t ofmap_dims[4], // output
    const uint64_t stride_dims[4],
    const ARBPRECTYPE dtype,
    POOLFUNC pool_func);

void
compile_resadd(FILE *out_binary,
    const addr_t lhs_addr,
    const addr_t rhs_addr,
    const uint64_t dims[4],
    const ARBPRECTYPE dtype);
*****************************************************************************/
        """
        )

        f.write(info)


    #-----------------------------------------------------------------
    def generateFile(self):
        nl = "\n"
        self.writeIfc()
        f = self.__File


        ####################################################
        ind = self.gIndent()
        sep = "//-----------------------------------------" + nl

        ####################################################
        header = [
            nl,
            "void",
            "network(const char* out_binary_name)",
            "{",
            ind + "FILE* const out_binary = fopen(out_binary_name);",
            ind + "uint64_t ofmap_dims[4];",
            ind + "uint64_t ifmap_dims[4];",
            ind + "uint64_t filter_dims[4];",
            ind + "uint64_t kernel_dims[4];",
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
            "}",
            "",
        ]
        for l in footer:
            f.write(l+nl)

