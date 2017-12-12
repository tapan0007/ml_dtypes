#include <cstdio>

using namespace std;

#include "network.hpp"
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"

#include "codegen.hpp"
#include "codegeninputlayer.hpp"
#include "codegenconvlayer.hpp"
#include "codegenrelulayer.hpp"
#include "codegentanhlayer.hpp"

namespace kcc {
using layers::InputLayer;
using layers::ConvLayer;
using layers::ReluLayer;
using layers::TanhLayer;

namespace codegen {


//  ##########################################################
//  void compile_read_ifmap(FILE *out_binary,
//          const addr_t ifmap_sb_addr, const char *in_numpy_fname,
//          const char *numpy_layout);
//
//  void compile_read_filter(FILE *out_binary,
//          const addr_t filter_sb_addr, const char *in_numpy_fname,
//          const char *numpy_layout);
//
//  void compile_write_ofmap(FILE *out_binary,
//          const char *out_numpy_name, const addr_t ofmap_sb_addr,
//          const uint64_t dims[4],
//          const ARBPRECTYPE dtype);
//
//  /*[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays
//   * deal with cases iwhen with the number of ifmap channels is > number of rows.
//   * In this case, the ifmaps and filters must be "wrapped".  Each address in the
//   * array is the wrap offset */
//  void
//  compile_convolve(FILE *out_binary,
//          const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], /* NCHW */
//          const addr_t *filter_addr, const uint64_t filter_dims[4], /* MCRS */
//          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
//          const ARBPRECTYPE in_dtype, const ARBPRECTYPE out_dtype,
//          const uint8_t padding[2],  /* Height,Width */
//          const uint8_t stride[2],   /* Height,Width */
//          const uint8_t dilate[2]);  /* Height,Width */
//
//  void
//  compile_pool(FILE *out_binary,
//          const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
//          const uint64_t kernel_dims[4], /* NCHW */
//          const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
//          const uint64_t stride_dims[4], /* NCHW */
//          const ARBPRECTYPE dtype,
//          POOLFUNC pool_func);
//  ##########################################################


//########################################################
CodeGen::CodeGen(Network* ntwk, Arch* arch)
{
    m_Network = ntwk;
    m_Arch = arch;
    createGenMap();
}

//----------------------------------------------------------------
void
CodeGen::createGenMap()
{
    m_InputLayer.reset(new CodeGenInputLayer(this));
    m_ConvLayer.reset(new CodeGenConvLayer(this));
    m_ReluLayer.reset(new CodeGenReluLayer(this));
    m_TanhLayer.reset(new CodeGenTanhLayer(this));
    //m_MaxPoolLayer.reset(new CodeGenMaxPoolLayer());
    //m_AvgPoolLayer.reset(new CodeGenAvgPoolLayer());
}

CodeGenLayer*
CodeGen::gGenFunc(const Layer* layer)
{
    if (dynamic_cast<const InputLayer*>(layer)) {
        return m_InputLayer.get();
    } else if (dynamic_cast<const ConvLayer*>(layer)) {
        return m_ConvLayer.get();
    } else if (dynamic_cast<const ReluLayer*>(layer)) {
        return m_ReluLayer.get();
    } else if (dynamic_cast<const TanhLayer*>(layer)) {
        return m_TanhLayer.get();
    } else {
        assert(false);
    }
}

void
CodeGen::generate(const char* objFileName)
{
    m_ObjFile = std::fopen(objFileName, "w");
    assert(m_ObjFile);

    for (auto layer : m_Network->gSchedForwLayers()) {
        CodeGenLayer* layerGen = gGenFunc(layer);
        layerGen->generate(layer);
    }

    fclose(m_ObjFile); m_ObjFile = nullptr;
}


#if 0
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
            else:
                raise

        ####################################################
        self.writeFooter(lastGenerator)
#endif

}}



