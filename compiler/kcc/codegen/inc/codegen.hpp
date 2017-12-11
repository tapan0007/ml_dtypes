#pragma once

#ifndef KCC_CODEGEN_CODEGEN_H
#define KCC_CODEGEN_CODEGEN_H

#include<string>
#include<memory>

using namespace std;

#include "layer.hpp"

namespace kcc {

namespace layers {
    class InputLayer;
    class ConvLayer;
}
namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}

namespace codegen {

using nets::Network;
using arch::Arch;
using layers::Layer;
using layers::InputLayer;
using layers::ConvLayer;

class CodeGenLayer;
class CodeGenInputLayer;
class CodeGenConvLayer;
class CodeGenReluLayer;
class CodeGenTanhLayer;

//########################################################
class CodeGen {
public:
    //----------------------------------------------------------------
    CodeGen(Network* ntwk, Arch* arch);

    //----------------------------------------------------------------
    FILE* gObjFile() const {
        return m_ObjFile;
    }


private:
    //----------------------------------------------------------------
    void createGenMap();

    //----------------------------------------------------------------
    string gDataTypeName(Layer* layer) const {
        string ret("ARBPRECTYPE::");
        ret += layer->gDataType()->gTccName();
        return  ret;
    }

    //----------------------------------------------------------------
    CodeGenLayer* gGenFunc(const Layer* layer);

    //----------------------------------------------------------------
    void generate(const char* fileName);


#if 0
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

private:
    FILE* m_ObjFile = nullptr;
    Network* m_Network = nullptr;
    Arch* m_Arch = nullptr;
    unique_ptr<CodeGenInputLayer> m_InputLayer;
    unique_ptr<CodeGenConvLayer>  m_ConvLayer;
    unique_ptr<CodeGenReluLayer> m_ReluLayer;
    unique_ptr<CodeGenTanhLayer>  m_TanhLayer;
};


}}

#endif // KCC_CODEGEN_CODEGEN_H

