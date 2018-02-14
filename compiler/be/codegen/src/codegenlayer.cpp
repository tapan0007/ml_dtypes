#include "nets/inc/network.hpp"

#include "codegen/inc/codegen.hpp"
#include "codegen/inc/codegenlayer.hpp"

namespace kcc {
namespace codegen {

//----------------------------------------------------------------
FILE*
CodeGenLayer::gObjFile() const
{
    return m_CodeGen->gObjFile();
}

//----------------------------------------------------------------
layers::Layer*
CodeGenLayer::gLayer() const
{
    return m_Layer;
}

void
CodeGenLayer::epilogue(const layers::Layer* const layer)
{
    if ((layer->gRefFileName() != "") || (layer->gNumNextLayers()==0)) {
        char outNpyFileName[256];

        if (layer->gRefFileName() != "") {
            sprintf(outNpyFileName, "%s", layer->gRefFileName().c_str());
            char* p = outNpyFileName + strlen(outNpyFileName);
            // skip last 4 chars ".npy"
            p -= 4;
            assert(0 == strcmp(p, ".npy") && "NPY filename suffix should be .npy");
            sprintf(p, "-simout.npy");
        } else {
            sprintf(outNpyFileName, "%s-%s-simout.npy",
                 layer->gNetwork()->gName().c_str(),
                 layer->gName().c_str());
        }

        for (char* p = outNpyFileName; *p; ++p) {
            if ('/' == *p) {
                *p = '-';
            }
        }

        FILE* const objFile = gObjFile();
        const ARBPRECTYPE outDataType = layer->gDataType().gSimTypeId();
        compile_write_ofmap(objFile,
                outNpyFileName,
                m_OfmapAddrs, m_OfmapDims, outDataType);
    }
}

}}

