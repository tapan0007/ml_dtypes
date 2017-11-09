
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

##  ##########################################################
##  void compile_read_ifmap(FILE *out_binary,
##          const addr_t ifmap_sb_addr, const char *in_numpy_fname,
##          const char *numpy_layout);
##
##  void compile_read_filter(FILE *out_binary,
##          const addr_t filter_sb_addr, const char *in_numpy_fname,
##          const char *numpy_layout);
##
##  void compile_write_ofmap(FILE *out_binary,
##          const char *out_numpy_name, const addr_t ofmap_sb_addr,
##          const uint64_t dims[4],
##          const ARBPRECTYPE dtype);
##
##  /*[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
##   * deal with cases iwhen with the number of ifmap channels is > number of rows.
##   * In this case, the ifmaps and filters must be "wrapped".  Each address in the
##   * array is the wrap offset */
##  void
##  compile_convolve(FILE *out_binary,
##          const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], /* NCHW */
##          const addr_t *filter_addr, const uint64_t filter_dims[4], /* MCRS */
##          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
##          const ARBPRECTYPE dtype,
##          const uint8_t padding[2],  /* Height,Width */
##          const uint8_t stride[2],   /* Height,Width */
##          const uint8_t dilate[2]);  /* Height,Width */
##
##  void
##  compile_pool(FILE *out_binary,
##          const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
##          const uint64_t kernel_dims[4], /* NCHW */
##          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
##          const uint64_t stride_dims[4], /* NCHW */
##          const ARBPRECTYPE dtype,
##          POOLFUNC pool_func);
##  ##########################################################



##########################################################
class MacroInstrGen(object):
    #-----------------------------------------------------------------
    def createGenMap(self):
        self.__Map = {
            layers.datalayer.DataLayer       : MacroDataLayer(self),
            layers.convlayer.ConvLayer       : MacroConvLayer(self),
            layers.maxpoollayer.MaxPoolLayer : MacroMaxPoolLayer(self),
            layers.avgpoollayer.AvgPoolLayer : MacroAvgPoolLayer(self),
            #layers.addlayer.AddLayer        : MacroAddLayer(self),
        }

    #-----------------------------------------------------------------
    def __init__(self, ntwk, arch):
        self.__Network = ntwk
        self.__Arch = arch
        self.__Indent = "    "
        self.__AssertionStr = "Assert"
        self.__FuncSep =  "//************************************************************"

        self.createGenMap()

    def gDataTypeName(self):
        return "ARBPRECTYPE::" + self.__Network.gDataType().gTccName()

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

    #-----------------------------------------------------------------
    def gFile(self):
        return self.__File

    #-----------------------------------------------------------------
    def writeIfc(self):
        f = self.__File

    def writeIncludes(self):
        f = self.__File
        f.write('\n')
        f.write('#include "tcc.h"\n')

    #-----------------------------------------------------------------
    def writeDefines(self):
        ind  = self.gIndent()
        f = self.__File
        f.write("#define Quote_(X) #X\n")
        f.write("#define Assert(X) assert((X) && Quote_(X))\n")
        f.write("enum FmapDimIndex {\n")

        f.write(ind+ "IfmapIndex_N = 0, // batch\n")
        f.write(ind+ "IfmapIndex_C = 1, // num ifmaps\n")
        f.write(ind+ "IfmapIndex_H = 2, // ifmap height\n")
        f.write(ind+ "IfmapIndex_W = 3, // ifmap width\n")

        f.write(ind+ "OfmapIndex_N = IfmapIndex_N, // batch\n")
        f.write(ind+ "OfmapIndex_C = IfmapIndex_C, // num ifmaps\n")
        f.write(ind+ "OfmapIndex_H = IfmapIndex_H, // ifmap height\n")
        f.write(ind+ "OfmapIndex_W = IfmapIndex_W, // ifmap width\n")

        f.write(ind+ "FilterIndex_M = 0, // filter num ofmaps\n")
        f.write(ind+ "FilterIndex_C = 1, // filter num ifmaps\n")
        f.write(ind+ "FilterIndex_R = 2, // filter height\n")
        f.write(ind+ "FilterIndex_S = 3, // filter width\n")
        f.write("};\n")

    #-----------------------------------------------------------------
    def writeFooter(self, lastGenerator):
        nl   = "\n"
        qq = '"'
        ind  = self.gIndent()
        ind2 = ind*2
        ind3 = ind*3
        sep  = self.__Sep
        f    = self.__File

        # void compile_write_ofmap(FILE *out_binary,
        #         const char *out_numpy_name, const addr_t ofmap_sb_addr,
        #         const uint64_t dims[4],
        #         const size_t word_size);
        assertStr  =  self.gAssertionStr()
        footer = [
            "",
            ind + "//-----------------------------------------------------",
            ind + "// output",
            ]

        #for s in lastGenerator.gWriteOfmapStatement(ind):
        #    footer.append(ind + s)
            
        footer += [
            ind + sep,
            ind + "return 0;",
            "}",
            "",

            self.__FuncSep,
            "class FileMgr {",
            "public:",
            ind + "FileMgr(FILE* file)",
            ind + "  : m_File(file)",
            ind + "{",
            ind2 +     assertStr +  "(file);",
            ind + "}",
            ind + "~FileMgr() {",
            ind2 +     "fclose(m_File);",
            ind + "}",
            "private:",
            ind + "FileMgr() = delete;",
            ind + "FileMgr(const FileMgr&) = delete;",
            "private:",
            ind + "FILE* const m_File;",
            "};",

            "",
            self.__FuncSep,
            "int",
            "main(int argc, char* argv[])",
            "{",
            ind + "if (argc < 2) {",
            ind2 +    'fprintf(stderr, "Usage: %s out_obj_file\\n", argv[0]);',
            ind2 +    "return 1;",
            ind + "}",
            ind + 'FILE* const out_binary = fopen(argv[1], "w");',

            ind + "if (! out_binary) {",
            ind2 +    'fprintf(stderr, "Cannot open file %s\\n", argv[1]);',
            ind2 +    "return 1;",
            ind + "}",
            ind + "FileMgr fileMgr(out_binary);",
            "",
            ind + "const int ret = " + self.__Network.gName() + "(out_binary);",
            ind + "if (ret != 0) {",
            ind2 +    'fprintf(stderr, "Usage: %s out_obj_file\\n", argv[0]);',
            ind + "}",
            ind + "return ret;",
            "}",
            "",
        ]

        for l in footer:
            f.write(l+nl)

    #-----------------------------------------------------------------
    def writeHeader(self):
        nl   = "\n"
        ind  = self.gIndent()
        ind2 = ind*2
        sep  = self.__Sep
        f    = self.__File

        header = [
            nl,
            self.__FuncSep,
            "static int",
            self.__Network.gName() + "(FILE* const out_binary)",
            "{",
            ind + "uint64_t ofmap_dims[4];",
            ind + "uint64_t ifmap_dims[4];",
            ind + "addr_t   ifmap_addrs[2] = {0, 0};",   ## 2 is temporary for single Ifmap
            ind + "addr_t   ofmap_addrs;",
            ind + "uint8_t  convolve_stride[2];",
            ind + "uint8_t  padding[2];",
            ind + "uint8_t  dilate[2];",
            "",
        ]

        qHasConvLayer = False
        qHasPoolLayer = False
        for layer in self.__Network.gSchedLayers():
            if layer.qConvLayer():
                qHasConvLayer = True
            elif layer.qPoolLayer():
                qHasPoolLayer = True

        if qHasConvLayer:
            header.append(ind + "uint64_t filter_dims[4];")
            header.append(ind + "addr_t   filter_addr[2] = {0, 0};")   ## 2 is temporary for single Ifmap
            header.append(ind + "const char* filter_file_names[2] = {0, 0};")   ## 2 is temporary for single Ifmap
            header.append("")

        if qHasPoolLayer:
            header.append(ind + "uint64_t kernel_dims[4];")
            header.append(ind + "uint64_t pool_stride[4];")

        for l in header:
            f.write(l+nl)

    #-----------------------------------------------------------------
    def generateFile(self):
        nl   = "\n"
        ind  = self.gIndent()
        ind2 = ind*2
        sep  = self.__Sep = "//-----------------------------------------" + nl
        f    = self.__File

        ####################################################
        self.writeIfc()
        self.writeIncludes()
        self.writeDefines()
        self.writeHeader()

        ####################################################
        lastGenerator = None
        lastLayer = None
        for layer in self.__Network.gSchedLayers():
            generator = self.gGenFunc(layer)
            if generator:
                f.write("\n" + ind + sep)
                generator.rLayer(layer)
                generator.generate()
                lastGenerator = generator

        ####################################################
        self.writeFooter(lastGenerator)

